use crate::trie_node::{NodeWithValue, NodeWithoutValue, TrieNode};
use std::{any::Any, fmt::Debug, sync::Arc};

#[derive(Clone, Default)]
pub struct Trie {
    root: Option<Arc<dyn TrieNode>>,
}

fn dfs(node: &Arc<dyn TrieNode>, key: &mut String, f: &mut std::fmt::DebugMap<'_, '_>) {
    if node.is_value_node() {
        f.entry(key, node);
    }

    for (&k, v) in node.children() {
        key.push(k);
        dfs(v, key, f);
        key.pop();
    }
}

impl Debug for Trie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.root.is_none() {
            return f.debug_struct("Trie").field("root", &"None").finish();
        }

        let mut key = String::new();
        let f = &mut f.debug_map();
        dfs(self.root.as_ref().unwrap(), &mut key, f);
        f.finish()
    }
}

impl Trie {
    fn with_root(root: Option<Arc<dyn TrieNode>>) -> Self {
        Self { root }
    }

    pub fn new() -> Self {
        Trie::default()
    }

    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        let cur = key
            .chars()
            .try_fold(self.root.as_ref()?, |cur, ch| cur.children().get(&ch))?;

        if cur.is_value_node() {
            let node = (cur.as_ref() as &dyn Any).downcast_ref::<NodeWithValue<T>>()?;
            Some(&node.value)
        } else {
            None
        }
    }

    pub fn put<T>(&self, key: &str, value: T) -> Self
    where
        T: Debug + Send + Sync + 'static,
    {
        let new_root = recursion_put(self.root.as_ref(), key, value);
        Self::with_root(Some(new_root.into()))
    }

    pub fn remove(&self, key: &str) -> Self {
        let new_root = if self.contains_key(key) {
            recursion_remove(self.root.as_ref().unwrap(), key)
        } else {
            self.root.clone()
        };
        Self::with_root(new_root)
    }

    fn contains_key(&self, key: &str) -> bool {
        self.root
            .as_ref()
            .and_then(|init| {
                key.chars()
                    .try_fold(init, |cur, ch| cur.children().get(&ch))
            })
            .map_or(false, |cur| cur.is_value_node())
    }
}

#[allow(clippy::indexing_slicing)]
fn recursion_remove(cur: &Arc<dyn TrieNode>, key: &str) -> Option<Arc<dyn TrieNode>> {
    if key.is_empty() {
        debug_assert!(cur.is_value_node());
        let has_remain_children = !cur.children().is_empty();
        return has_remain_children
            .then(|| NodeWithoutValue::with_children(cur.children().clone()).into());
    }

    let ch = &key.chars().next().unwrap();
    let new_child = recursion_remove(&cur.children()[ch], &key[1..]);

    if new_child.is_none() && cur.children().len() == 1 && !cur.is_value_node() {
        return None;
    }

    let mut new_root = cur.clone_node();
    if let Some(new_child) = new_child {
        *new_root.children_mut().get_mut(ch).unwrap() = new_child;
    } else {
        new_root.children_mut().remove(ch).unwrap();
    }
    Some(new_root.into())
}

#[allow(clippy::indexing_slicing)]
fn recursion_put<T>(cur: Option<&Arc<dyn TrieNode>>, key: &str, value: T) -> Box<dyn TrieNode>
where
    T: Debug + Send + Sync + 'static,
{
    if key.is_empty() {
        if let Some(cur) = cur {
            return NodeWithValue::with_children(cur.children().clone(), Arc::new(value));
        }
        return NodeWithValue::new_box(Arc::new(value));
    }

    let ch = key.chars().next().unwrap();
    let new_child = recursion_put(
        cur.and_then(|node| node.children().get(&ch)),
        &key[1..],
        value,
    );

    let mut new_root = cur.map_or_else(NodeWithoutValue::new_box, |node| node.clone_node());
    new_root.children_mut().insert(ch, new_child.into());

    new_root
}
