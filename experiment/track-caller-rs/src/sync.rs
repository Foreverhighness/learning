mod refcell;

pub use refcell::SyncRefCell;

/// Alias type of SyncRefCell
pub type RefCell<T> = refcell::SyncRefCell<T>;
