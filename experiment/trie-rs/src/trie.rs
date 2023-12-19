use crate::trie_node::{TrieNode, TrieNodeWithValue, TrieNodeWithoutValue};
use std::{any::Any, fmt::Debug, sync::Arc};

#[derive(Clone, Default)]
pub struct Trie {
    root: Option<Arc<dyn TrieNode>>,
}

fn dfs(node: &Arc<dyn TrieNode>, key: &mut String, f: &mut std::fmt::Formatter<'_>) {
    if node.is_value_node() {
        let _ = writeln!(f, "{{{key}: {node:?}}}");
    }

    for (&k, v) in node.children() {
        key.push(k);
        dfs(v.as_ref().unwrap(), key, f);
        key.pop();
    }
}

impl Debug for Trie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.root.is_none() {
            return f.debug_struct("Trie").field("root", &"None").finish();
        }
        let mut key = String::new();
        dfs(self.root.as_ref().unwrap(), &mut key, f);
        Ok(())
    }
}

impl Trie {
    fn with_root(root: Option<Arc<dyn TrieNode>>) -> Self {
        Self { root }
    }

    pub fn new() -> Self {
        Default::default()
    }

    pub fn get<T: 'static>(&self, key: &str) -> Option<&T> {
        let mut cur = self.root.as_ref()?;
        for ch in key.chars() {
            cur = cur.children().get(&ch)?.as_ref().expect("invalid child");
        }

        if cur.is_value_node() {
            let node = (cur.as_ref() as &dyn Any).downcast_ref::<TrieNodeWithValue<T>>()?;
            Some(&node.value)
        } else {
            None
        }
    }

    pub fn put<T>(&self, key: &str, value: T) -> Self
    where
        T: Debug + Send + Sync + 'static,
    {
        let mut ret = Self::with_root(self.root.clone());
        recursion_put(&mut ret.root, key, value);
        ret
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
        let Some(mut cur) = self.root.as_ref() else {
            return false;
        };
        for ch in key.chars() {
            let Some(nxt) = cur.children().get(&ch) else {
                return false;
            };
            cur = nxt.as_ref().unwrap();
        }
        cur.is_value_node()
    }
}

fn recursion_remove(cur: &Arc<dyn TrieNode>, key: &str) -> Option<Arc<dyn TrieNode>> {
    if key.is_empty() {
        debug_assert!(cur.is_value_node());
        let has_remain_children = !cur.children().is_empty();
        return has_remain_children
            .then(|| TrieNodeWithoutValue::with_children(cur.children().clone()).into());
    }

    let ch = &key.chars().next().unwrap();
    let new_child = recursion_remove(cur.children()[ch].as_ref().unwrap(), &key[1..]);

    if new_child.is_none() && cur.children().len() == 1 && !cur.is_value_node() {
        return None;
    }

    let mut new_root = TrieNode::clone(&**cur);
    if new_child.is_none() {
        new_root.children_mut().remove(ch).unwrap();
    } else {
        *new_root.children_mut().get_mut(ch).unwrap() = new_child;
    }
    Some(new_root.into())
}

fn recursion_put<T>(cur: &mut Option<Arc<dyn TrieNode>>, key: &str, value: T)
where
    T: Debug + Send + Sync + 'static,
{
    if key.is_empty() {
        let node = {
            if let &mut Some(ref cur) = cur {
                TrieNodeWithValue::with_children(cur.children().clone(), Arc::new(value))
            } else {
                TrieNodeWithValue::new_box(Arc::new(value))
            }
        };
        *cur = Some(node.into());
        return;
    }

    let ch = key.chars().next().unwrap();

    let mut copy = cur.as_ref().map_or(TrieNodeWithoutValue::new_box(), |cur| {
        TrieNode::clone(&**cur)
    });

    recursion_put(copy.children_mut().entry(ch).or_default(), &key[1..], value);

    *cur = Some(copy.into());
}
