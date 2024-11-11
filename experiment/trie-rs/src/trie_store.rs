use std::ptr::NonNull;
use std::sync::{Mutex, RwLock};

use crate::trie::Trie;
use crate::trie_node::NodeValueMarkerTrait;

pub struct ValueGuard<'v, T>
where
    T: NodeValueMarkerTrait,
{
    _root: Trie,
    value: &'v T,
}

impl<T> ValueGuard<'_, T>
where
    T: NodeValueMarkerTrait,
{
    const fn new(root: Trie, value: NonNull<T>) -> Self {
        // SAFETY: caller safety guarantees
        let value = unsafe { value.as_ref() };
        Self { _root: root, value }
    }
}

impl<T> std::ops::Deref for ValueGuard<'_, T>
where
    T: NodeValueMarkerTrait,
{
    type Target = T;

    fn deref(&self) -> &Self::Target {
        self.value
    }
}

#[derive(Debug, Default)]
pub struct TrieStore {
    root: RwLock<Trie>,
    write_lock: Mutex<()>,
}

impl TrieStore {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn get<T>(&self, key: &str) -> Option<ValueGuard<T>>
    where
        T: NodeValueMarkerTrait,
    {
        let root = self.root.read().unwrap().clone();

        let value = NonNull::from(root.get(key)?);
        Some(ValueGuard::new(root, value))
    }

    pub fn put<T>(&self, key: &str, value: T)
    where
        T: NodeValueMarkerTrait,
    {
        let _guard = self.write_lock.lock().unwrap();

        let root = self.root.read().unwrap().clone();

        *self.root.write().unwrap() = root.put(key, value);
    }

    pub fn remove(&self, key: &str) {
        let _guard = self.write_lock.lock().unwrap();

        let root = self.root.read().unwrap().clone();

        *self.root.write().unwrap() = root.remove(key);
    }
}
