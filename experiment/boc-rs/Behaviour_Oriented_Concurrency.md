## 

Main thread is considered as root behavior.

`when` is response to spawn new thread `routine`.

## Synchronization



## Lock-free

In `StartAppendRequest`, we atomic swap the new tail to the old, so it is guarantee to process.

Look at `main`, we are not going to take `r1`'s ownership, because we need to transfer many instance to multiple when clause.



In `main`, `when` will immediately schedule an instance of `Behavior`, by using `Behavior::new()`.  
In `main`, we create three certificates of resource 1, named `r1c1`, `r1c2`, `r1c3`.  
These three certificate shared one pointed area, to indicate the last certificate.

```rust
struct ResourceCertificate {
    resource: Weak<Resource>,
    last_request: AtomicPtr<Request>,
}
struct ResourceOwned {
    resource: Arc<Resource>,
    last_request: AtomicPtr<Request>,
}
struct ResourceUsed;
struct Request {
    resource: Enum(ResourceCertificate, ResourceOwned, ResourceUsed),
    /// Behavior is use to poll next behavior
    /// Option is state of behavior
    next: AtomicPtr<Behavior>,
    scheduled: AtomicBool,
}
struct Behavior {
    routine: impl FnOnce() + Send,
    count: WaitGroup,
    requests: Box<[Request]>,
}
impl Behavior {
    pub fn new<F>(routine: F, resources: Vec<ResourceCertificate>) {
        Self {
            routine,
            count: resources.len() + 1,
            requests: resources.into_iter().map(Request::new).collect()
        }
    }
}
```

```rust
/// In real implementation, there is no explicit scheduler
/// so we are not encounter one point contention
impl ImplicitScheduler { // or impl Runtime
    /// schedule will blocking `when` clause, so there is guarantee to convert one Weak to Arc
    pub fn schedule<F>(routine: F, mut resources: Vec<ResourceCertificate>) {
        resources.sort();

        /// behavior may store on stack, which may not live so long
        let mut behavior = Behavior::new(routine, resources);

        for index, req in behavior.request {
            assert!(req.is_certificate());
            assert!(!req.is_owned());

            let prev_req;
            // always success, lock-free
            atomic { prev_req <- req.resource.last_request <- req; } // Relaxed

            // prev_req == nullptr means that no one has ownership
            if prev_req.is_none() {
                req.to_owned(); // Relaxed
                continue;
            }

            /// change schedule to true is pretty quick, so busy loop here
            while !prev_req.scheduled {
                core::hint::spin_loop();
            }

            assert!(prev_req.scheduled)

            prev_req.next = behavior;
            prev_req.target_index = index

            prev_req.transmitter.link_to(behavior.receiver)
        }

        for req in behavior.request {
            req.scheduled = true;
        }

        Scheduler::add_task(|| move {
            while !behavior.request.all(|req| req.is_owned()) {
                let index, owned_resource <- behavior.receiver
                assert!(behavior.request[index].is_certificate())
                behavior.request[index] = owned_resource
            }
            assert!(behavior.request.all(|req| req.is_owned()));

            behavior.routine();

            for req in behavior.requests {
                let owned_resource = req.take_ownership();

                if req.next.is_none() {
                    atomic {
                        let last_request = req.resource.last_request;
                        if last_request == req {
                            req.resource.last_request = None;
                            return
                        }
                        last_request
                    }

                    while req.next.is_none() {
                        core::hint::spin_loop();
                    }
                }
                // req.next <- req.target_index, owned_resource
                req.transmitter <- req.target_index, owned_resource
            }
        });
    }
}
```

```rust
fn main() {
    when!(r1,       ; || { /* b0 */ });
    when!(        r3; || { /* b1 */ });
    when!(r1, r2    ; || { /* b2 */ });
    when!(r1,       ; || { /* b3 */ });
    when!(    r2, r3; || { /* b4 */ });
    when!(        r3; || { /* b5 */ });
    wait();
}
```

## Relative to distribute system

In the


[Verona runtime]: https://github.com/microsoft/verona-rt
