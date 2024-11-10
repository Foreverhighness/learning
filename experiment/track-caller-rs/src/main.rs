use std::rc::Rc;

use track_caller_rs::sync::SyncRefCell;

fn main() {
    let a = Rc::new(SyncRefCell::new(1));
    let _b = a.borrow();
    let _c = a.borrow_mut();
}
