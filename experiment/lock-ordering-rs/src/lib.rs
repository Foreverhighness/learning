//! Inspired by <https://www.youtube.com/watch?v=Ba7fajt4l1M>

use core::marker::PhantomData;
use std::sync::{MutexGuard, PoisonError};

// region:marker

pub trait LockAfter<A> {}

pub trait LockBefore<Z> {}

impl<A, Z> LockBefore<Z> for A where Z: LockAfter<A> {}

// endregion:marker

// region:LockContext

pub struct LockContext<Id>(PhantomData<Id>);

pub struct Unlocked;

#[must_use]
pub const fn new_lock_context() -> LockContext<Unlocked> {
    LockContext::new()
}

impl<Id> LockContext<Id> {
    const fn new() -> Self {
        Self(PhantomData)
    }
}

// endregion: LockContext

// region:Mutex

pub struct OrderedMutex<Id, T>(std::sync::Mutex<T>, LockContext<Id>);
pub type LockResult<Id, T> = Result<(T, LockContext<Id>), PoisonError<T>>;

impl<Z, T> OrderedMutex<Z, T> {
    /// # Errors
    ///
    /// If another user of this mutex panicked while holding the mutex, then
    /// this call will return an error once the mutex is acquired. The acquired
    /// mutex guard will be contained in the returned error.
    pub fn lock<A>(&self, _: &mut LockContext<A>) -> LockResult<Z, MutexGuard<T>>
    where
        A: LockBefore<Z>,
    {
        self.0.lock().map(|guard| (guard, LockContext::new()))
    }
}

// endregion:Mutex

// region:example

/// Example Ordering
///
/// ```none
/// A───►B────┐
/// │    │    │
/// │    ▼    ▼
/// └───►C───►D
/// ```
pub mod example {
    use super::{LockAfter, Unlocked};

    pub struct A;
    pub struct B;
    pub struct C;
    pub struct D;

    impl LockAfter<Unlocked> for A {}
    impl LockAfter<Unlocked> for B {}
    impl LockAfter<Unlocked> for C {}
    impl LockAfter<Unlocked> for D {}

    impl LockAfter<A> for B {}
    impl LockAfter<A> for C {}

    impl LockAfter<B> for C {}
    impl LockAfter<B> for D {}

    impl LockAfter<C> for D {}
}

// endregion:example

#[cfg(test)]
mod tests {
    #[test]
    fn compile_fail() {
        let t = trybuild::TestCases::new();
        t.pass("tests/pass/*.rs");
        t.compile_fail("tests/fail/*.rs");
    }
}
