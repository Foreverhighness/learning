use crate::trie_node::{TrieNode, TrieNodeWithValue, TrieNodeWithoutValue};
use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

#[derive(Clone, Default)]
pub struct Trie {
    root: Option<Arc<dyn TrieNode>>,
}

impl Debug for Trie {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.root.is_none() {
            return f.debug_struct("Trie").field("root", &"None").finish();
        }

        fn dfs(node: &Arc<dyn TrieNode>, key: &mut String, map: &mut HashMap<String, String>) {
            if node.is_value_node() {
                map.insert(key.clone(), format!("{node:?}"));
            }

            for (&k, v) in node.children() {
                key.push(k);
                dfs(v, key, map);
                key.pop();
            }
        }

        let mut map = HashMap::new();
        let mut key = String::new();
        dfs(self.root.as_ref().unwrap(), &mut key, &mut map);
        f.debug_map().entries(map).finish()
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
            cur = cur.children().get(&ch)?;
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
        let Some(mut cur) = self.root.as_ref() else {
            return false;
        };
        for ch in key.chars() {
            let Some(nxt) = cur.children().get(&ch) else {
                return false;
            };
            cur = nxt;
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
    let new_child = recursion_remove(&cur.children()[ch], &key[1..]);

    if new_child.is_none() && cur.children().len() == 1 && !cur.is_value_node() {
        return None;
    }

    let mut new_root = TrieNode::clone(&**cur);
    if let Some(new_child) = new_child {
        *new_root.children_mut().get_mut(ch).unwrap() = new_child;
    } else {
        new_root.children_mut().remove(ch).unwrap();
    }
    Some(new_root.into())
}

fn recursion_put<T>(cur: Option<&Arc<dyn TrieNode>>, key: &str, value: T) -> Box<dyn TrieNode>
where
    T: Debug + Send + Sync + 'static,
{
    if key.is_empty() {
        if let Some(cur) = cur {
            return TrieNodeWithValue::with_children(cur.children().clone(), Arc::new(value));
        } else {
            return TrieNodeWithValue::new_box(Arc::new(value));
        };
    }

    let ch = key.chars().next().unwrap();
    let new_child = recursion_put(
        cur.and_then(|node| node.children().get(&ch)),
        &key[1..],
        value,
    );

    let mut new_root = cur.map_or_else(TrieNodeWithoutValue::new_box, |node| {
        TrieNode::clone(&**node)
    });
    new_root.children_mut().insert(ch, new_child.into());

    new_root
}
