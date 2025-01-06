use lock_ordering_rs::example::*;
use lock_ordering_rs::{OrderedMutex, new_lock_context};

#[expect(unused, reason = "testing")]
fn fail(a: OrderedMutex<A, ()>, d: OrderedMutex<D, ()>) {
    let mut ctx = new_lock_context();
    let (a, mut ctx) = a.lock(&mut ctx).unwrap();
    let (d, mut ctx) = d.lock(&mut ctx).unwrap();
}

fn main() {}
