//! Behavior of concurrency
#[expect(clippy::borrow_deref_ref)]
#[allow(dead_code)]
pub mod basic {
    use core::fmt::Debug;
    use core::hint;
    use core::marker::PhantomData;
    use core::mem::ManuallyDrop;
    use core::pin::Pin;
    use core::ptr::{self, NonNull};
    use core::sync::atomic::{AtomicBool, AtomicPtr, AtomicUsize, Ordering};
    use std::sync::Arc;

    use crate::runtime;

    // region:Cown

    /// A trait representing a `Cown`.
    ///
    /// Instead of directly using a `Cown<T>`, which fixes _a single_ `T` we use a trait object to
    /// allow multiple requests with different `T`s to be used with the same cown.
    trait CownTrait {
        /// this method restrict that Request should not contains any generic?
        fn last(&self) -> &AtomicPtr<Request>;

        /// substitute last method to not expose AtomicPtr
        fn last_swap(&self, ptr: *mut Request, order: Ordering) -> *mut Request {
            self.last().swap(ptr, order)
        }

        /// substitute last method to not expose AtomicPtr
        fn last_compare_exchange(
            &self,
            current: *mut Request,
            new: *mut Request,
            success: Ordering,
            failure: Ordering,
        ) -> Result<*mut Request, *mut Request> {
            self.last().compare_exchange(current, new, success, failure)
        }
    }

    // /// Use RefCell to check correctness, but failed to pass borrow checker
    // type InteriorMutCell<T> = core::cell::RefCell<T>;
    // type CownRefMut<'l, T> = core::cell::RefMut<'l, T>;

    type InteriorMutCell<T> = core::cell::UnsafeCell<T>;
    // type CownRefMut<'l, T> = &'l mut T;

    /// Concurrency Owned Resource
    /// The value should only be accessed inside a `when!` block.
    #[derive(Debug)]
    struct Cown<T: ?Sized + 'static> {
        /// MCS lock tail.
        ///
        /// When a new node is enqueued, the enqueuer of the previous tail node will wait until the
        /// current enqueuer sets that node's `.next`.
        last: AtomicPtr<Request>,

        /// The value of this cown.
        resource: InteriorMutCell<T>,
    }

    impl<T> Cown<T> {
        const fn new(value: T) -> Self {
            Self {
                last: AtomicPtr::new(ptr::null_mut()),
                resource: InteriorMutCell::new(value),
            }
        }
    }

    /// Send bound is copied from `std::sync::Mutex`
    /// Behavior must ensure there is only one thread modify Cown
    unsafe impl<T: Send> Sync for Cown<T> {}

    impl<T> CownTrait for Cown<T> {
        fn last(&self) -> &AtomicPtr<Request> {
            &self.last
        }
    }

    /// Hold Cown to access last
    struct CownHolder {
        /// use to access last ptr, but need extend resource's lifetime
        inner: Arc<dyn CownTrait>,

        /// effectively we only concern about this
        _last: PhantomData<Arc<AtomicPtr<Behavior>>>,
    }

    /// Safety: We only use CownHolder to access AtomicPtr
    unsafe impl Send for CownHolder {}

    impl Debug for CownHolder {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("CownHolder")
                .field("cown", &Arc::as_ptr(&self.inner))
                .field("last", &self.inner.last())
                .finish()
        }
    }
    impl CownTrait for CownHolder {
        fn last(&self) -> &AtomicPtr<Request> {
            self.inner.last()
        }
    }

    impl CownHolder {
        fn new<T: 'static + Send>(cown: CownPtr<T>) -> Self {
            Self {
                inner: cown.inner,
                _last: PhantomData,
            }
        }
    }

    /// Public interface to Cown.
    #[derive(Debug)]
    pub struct CownPtr<T: 'static> {
        inner: Arc<Cown<T>>,
    }

    impl<T: 'static> CownPtr<T> {
        pub fn new(value: T) -> Self {
            Self {
                inner: Arc::new(Cown::new(value)),
            }
        }
    }

    impl<T: Send + 'static> Clone for CownPtr<T> {
        fn clone(&self) -> Self {
            Self {
                inner: Arc::clone(&self.inner),
            }
        }
    }

    // region:ArrayOfCownPtr

    /// Trait for a collection of `CownPtr`s.
    ///
    /// Users pass `CownPtrs` to `when!` clause to specify a collection of shared resources, and
    /// such resources can be accessed via `CownRefs` inside the thunk.
    pub trait CownPtrs {
        /// Types for references corresponding to `CownPtrs`.
        type CownRefs<'l>
        where
            Self: 'l;

        // This could return a `Box<[Request]>`, but we use a `Vec` to avoid possible reallocation
        // in the implementation.
        /// Returns a collection of `Request`.
        fn requests(&self) -> Vec<Request>;

        /// Returns mutable references of type `CownRefs`.
        fn get_mut<'l>(self) -> Self::CownRefs<'l>;
    }

    impl CownPtrs for () {
        type CownRefs<'l>
            = ()
        where
            Self: 'l;

        fn requests(&self) -> Vec<Request> {
            Vec::new()
        }

        fn get_mut<'l>(self) -> Self::CownRefs<'l> {}
    }

    impl<T: Send + 'static, Ts: CownPtrs> CownPtrs for (CownPtr<T>, Ts) {
        type CownRefs<'l>
            = (&'l mut T, Ts::CownRefs<'l>)
        where
            Self: 'l;

        fn requests(&self) -> Vec<Request> {
            let mut rs = self.1.requests();
            rs.push(Request::new(CownPtr::clone(&self.0)));
            rs
        }

        fn get_mut<'l>(self) -> Self::CownRefs<'l> {
            unsafe { (&mut *self.0.inner.resource.get(), self.1.get_mut()) }
        }
    }

    impl<T: Send + 'static> CownPtrs for Vec<CownPtr<T>> {
        type CownRefs<'l>
            = Vec<&'l mut T>
        where
            Self: 'l;

        fn requests(&self) -> Vec<Request> {
            self.iter()
                .map(|x| Request::new(CownPtr::clone(x)))
                .collect()
        }

        fn get_mut<'l>(self) -> Self::CownRefs<'l> {
            self.iter()
                .map(|x| unsafe { &mut *x.inner.resource.get() })
                .collect()
        }
    }

    // endregion:ArrayOfCownPtr

    // endregion:Cown

    // region:Request

    #[derive(Debug)]
    pub struct Request {
        /// use to call next resolve_one()
        next: AtomicPtr<Behavior>,

        /// two phase locking
        scheduled: AtomicBool,

        /// access cown last, ensure cown is valid
        target: CownHolder,
    }

    impl Request {
        /// Creates a new Request.
        fn new<T: Send + 'static>(target: CownPtr<T>) -> Request {
            Request {
                next: AtomicPtr::new(ptr::null_mut()),
                scheduled: AtomicBool::new(false),
                target: CownHolder::new(target),
            }
        }

        /// start_enqueue can executed parallel, so we need shared ref on behavior
        /// but the self ref may be exclusive?
        fn start_enqueue(&self, behavior: &Behavior) {
            let prev_req = self
                .target
                .last_swap((&raw const *self).cast_mut(), Ordering::Relaxed);

            let Some(prev_req) = (unsafe { prev_req.as_ref() }) else {
                // prev_req is null
                behavior.resolve_one();
                return;
            };

            while !prev_req.scheduled.load(Ordering::Relaxed) {
                hint::spin_loop();
            }

            debug_assert!(prev_req.next.load(Ordering::Relaxed).is_null());
            prev_req
                .next
                .store((&raw const *behavior).cast_mut(), Ordering::Relaxed);
        }

        /// Finish the second phase of the 2PL enqueue operation.
        ///
        /// Sets the scheduled flag so that subsequent behaviors can continue the 2PL enqueue.
        fn finish_enqueue(&self) {
            self.scheduled.store(true, Ordering::Relaxed);
        }

        fn release(&self) {
            let mut behavior = self.next.load(Ordering::Relaxed);
            if behavior.is_null() {
                if self
                    .target
                    .last_compare_exchange(
                        (&raw const *self).cast_mut(),
                        ptr::null_mut(),
                        Ordering::Relaxed,
                        Ordering::Relaxed,
                    )
                    .is_ok()
                {
                    return;
                }

                loop {
                    behavior = self.next.load(Ordering::Relaxed);
                    if !behavior.is_null() {
                        break;
                    }
                    core::hint::spin_loop();
                }
            }

            let behavior = unsafe { behavior.as_ref().unwrap() };
            behavior.resolve_one();

            self.next.store(ptr::null_mut(), Ordering::Relaxed);
        }
    }

    impl Ord for Request {
        fn cmp(&self, other: &Self) -> core::cmp::Ordering {
            Arc::as_ptr(&self.target.inner)
                .addr()
                .cmp(&Arc::as_ptr(&other.target.inner).addr())
        }
    }
    impl PartialOrd for Request {
        fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
            Some(self.cmp(other))
        }
    }
    impl PartialEq for Request {
        fn eq(&self, other: &Self) -> bool {
            matches!(self.cmp(other), core::cmp::Ordering::Equal)
        }
    }
    impl Eq for Request {}

    // endregion:Request

    // region:Behavior

    pub struct Behavior {
        routine: Box<dyn FnOnce() + Send>,
        count: AtomicUsize,
        requests: Pin<Box<[Request]>>,
    }

    impl Debug for Behavior {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.debug_struct("Behavior")
                .field("count", &self.count)
                .field("requests", &self.requests)
                .finish()
        }
    }

    impl Behavior {
        fn schedule(&self) {
            // eprintln!("schedule start: addr {:?} {self:?}", &raw const *self);
            debug_assert!(self.requests.is_sorted());

            self.requests.iter().for_each(|req| req.start_enqueue(self));

            self.requests.iter().for_each(|req| req.finish_enqueue());

            self.resolve_one();
            // eprintln!("schedule over: addr {:?} {self:?}", &raw const *self);
        }

        pub fn resolve_one(&self) {
            if self.count.fetch_sub(1, Ordering::Relaxed) > 1 {
                return;
            }
            debug_assert_eq!(self.count.load(Ordering::Relaxed), 0);

            unsafe { PinnedBehavior::from_inner(NonNull::from_ref(self)) }.run();
        }
    }

    pub struct PinnedBehavior(Pin<Box<Behavior>>);

    impl PinnedBehavior {
        /// new Behavior at heap
        pub fn new<Cs, F>(cowns: Cs, f: F) -> Self
        where
            Cs: CownPtrs + Send + 'static,
            F: for<'l> FnOnce(Cs::CownRefs<'l>) + Send + 'static,
        {
            let mut requests = cowns.requests();
            requests.sort();

            let requests = Pin::new(requests.into_boxed_slice());
            let count = AtomicUsize::new(requests.len() + 1);

            let routine = Box::new(move || f(cowns.get_mut()));
            PinnedBehavior(Box::pin(Behavior {
                routine,
                count,
                requests,
            }))
        }

        /// schedule a behavior
        pub fn schedule(self) -> ManuallyDrop<Self> {
            self.0.schedule();

            ManuallyDrop::new(self)
        }

        /// submit behavior to runtime
        fn run(self) {
            // eprintln!("start running");
            let behavior = Pin::into_inner(self.0);
            runtime::spawn(move || {
                (behavior.routine)();

                behavior.requests.iter().for_each(|req| req.release());
            });
        }

        unsafe fn from_inner(ptr: NonNull<Behavior>) -> Self {
            Self(Box::into_pin(unsafe { Box::from_raw(ptr.as_ptr()) }))
        }
    }

    // endregion:Behavior

    // region:when

    /// Creates a `Behavior` and schedules it. Used by "When" block.
    pub fn run_when<C, F>(cowns: C, f: F)
    where
        C: CownPtrs + Send + 'static,
        F: for<'l> Fn(C::CownRefs<'l>) + Send + 'static,
    {
        PinnedBehavior::new(cowns, f).schedule();
    }

    /// from <https://docs.rs/tuple_list/latest/tuple_list/>
    #[macro_export]
    macro_rules! tuple_list {
        () => ( () );

        // handling simple identifiers, for limited types and patterns support
        ($i:ident)  => ( ($i, ()) );
        ($i:ident,) => ( ($i, ()) );
        ($i:ident, $($e:ident),*)  => ( ($i, $crate::tuple_list!($($e),*)) );
        ($i:ident, $($e:ident),*,) => ( ($i, $crate::tuple_list!($($e),*)) );

        // handling complex expressions
        ($i:expr)  => ( ($i, ()) );
        ($i:expr,) => ( ($i, ()) );
        ($i:expr, $($e:expr),*)  => ( ($i, $crate::tuple_list!($($e),*)) );
        ($i:expr, $($e:expr),*,) => ( ($i, $crate::tuple_list!($($e),*)) );
    }

    /// "When" block.
    #[macro_export]
    macro_rules! when {
        ( $( $cs:ident ),* ; $( $gs:ident ),* ; $thunk:expr ) => {{
            run_when($crate::tuple_list!($($cs.clone()),*), move |$crate::tuple_list!($($gs),*)| $thunk);
        }};
    }

    // endregion:when
}

#[cfg(test)]
mod tests {
    use crate::*;

    #[test]
    fn boc() {
        let c1 = CownPtr::new(0);
        let c2 = CownPtr::new(0);
        let c3 = CownPtr::new(false);
        let c2_ = c2.clone();
        let c3_ = c3.clone();

        let (finish_sender, finish_receiver) = bounded(0);

        when!(c1, c2; g1, g2; {
            // c3, c2 are moved into this thunk. There's no such thing as auto-cloning move closure.
            *g1 += 1;
            *g2 += 1;
            when!(c3, c2; g3, g2; {
                *g2 += 1;
                *g3 = true;
            });
        });

        when!(c1, c2_, c3_; g1, g2, g3; {
            assert_eq!(*g1, 1);
            assert_eq!(*g2, if *g3 { 2 } else { 1 });
            finish_sender.send(()).unwrap();
        });

        // wait for termination
        finish_receiver.recv().unwrap();
    }

    #[test]
    fn boc_vec() {
        let c1 = CownPtr::new(0);
        let c2 = CownPtr::new(0);
        let c3 = CownPtr::new(false);
        let c2_ = c2.clone();
        let c3_ = c3.clone();

        let (finish_sender, finish_receiver) = bounded(0);

        run_when(vec![c1.clone(), c2.clone()], move |mut x| {
            // c3, c2 are moved into this thunk. There's no such thing as auto-cloning move closure.
            *x[0] += 1;
            *x[1] += 1;
            when!(c3, c2; g3, g2; {
                *g2 += 1;
                *g3 = true;
            });
        });

        when!(c1, c2_, c3_; g1, g2, g3; {
            assert_eq!(*g1, 1);
            assert_eq!(*g2, if *g3 { 2 } else { 1 });
            finish_sender.send(()).unwrap();
        });

        // wait for termination
        finish_receiver.recv().unwrap();
    }

    #[test]
    fn concurrency() {
        let num_thread = 2;
        let count = 100_000_000u64;

        let (tx1, rx) = bounded(num_thread);
        let tx2 = tx1.clone();

        let c1 = CownPtr::new(0);
        let c2 = CownPtr::new(0);

        let calc = || core::hint::black_box(1);

        when!(c1; g1; {
            for _ in 0..core::hint::black_box(count) {
                *g1 += core::hint::black_box(calc());
            }
            tx1.send(*g1).unwrap();
        });

        when!(c2; g2; {
            for _ in 0..core::hint::black_box(count) {
                *g2 += core::hint::black_box(calc());
            }
            tx2.send(*g2).unwrap();
        });

        for _ in 0..num_thread {
            let ret = rx.recv().unwrap();
            assert_eq!(ret, count);
        }
    }
}
