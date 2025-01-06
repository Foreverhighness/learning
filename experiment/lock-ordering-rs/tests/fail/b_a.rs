use lock_ordering_rs::example::*;
use lock_ordering_rs::{OrderedMutex, new_lock_context};

#[expect(unused, reason = "testing")]
fn fail(a: OrderedMutex<A, ()>, b: OrderedMutex<D, ()>) {
    let mut ctx = new_lock_context();
    let (b, mut ctx) = b.lock(&mut ctx).unwrap();
    let (a, mut ctx) = a.lock(&mut ctx).unwrap();
}

fn main() {}
