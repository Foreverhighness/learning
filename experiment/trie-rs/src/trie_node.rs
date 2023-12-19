use std::{any::Any, collections::HashMap, fmt::Debug, sync::Arc};

type Children = HashMap<char, Arc<dyn TrieNode>>;

pub trait TrieNode: Debug + Send + Sync + Any {
    fn children(&self) -> &Children;
    fn children_mut(&mut self) -> &mut Children;
    fn is_value_node(&self) -> bool;
    fn clone_node(&self) -> Box<dyn TrieNode>;
}

#[derive(Clone, Default)]
pub struct TrieNodeWithoutValue {
    children: Children,
    is_value_node: bool,
}

impl Debug for TrieNodeWithoutValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl TrieNodeWithoutValue {
    pub fn new_box() -> Box<dyn TrieNode> {
        Box::<TrieNodeWithoutValue>::default()
    }

    pub fn with_children(children: Children) -> Box<dyn TrieNode> {
        Box::new(Self {
            children,
            is_value_node: false,
        })
    }
}

impl TrieNode for TrieNodeWithoutValue {
    fn children(&self) -> &Children {
        &self.children
    }

    fn is_value_node(&self) -> bool {
        self.is_value_node
    }

    fn clone_node(&self) -> Box<dyn TrieNode> {
        Box::new(self.clone())
    }

    fn children_mut(&mut self) -> &mut Children {
        &mut self.children
    }
}

#[derive(Default)]
pub struct TrieNodeWithValue<T> {
    node: TrieNodeWithoutValue,
    pub value: Arc<T>,
}

impl<T: Debug> Debug for TrieNodeWithValue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

impl<T> TrieNodeWithValue<T>
where
    T: Debug + Send + Sync + 'static,
{
    pub fn new_box(value: Arc<T>) -> Box<dyn TrieNode> {
        Self::with_children(Default::default(), value)
    }

    pub fn with_children(children: Children, value: Arc<T>) -> Box<dyn TrieNode> {
        let node = TrieNodeWithoutValue {
            is_value_node: true,
            children,
        };
        Box::new(Self { node, value })
    }
}

impl<T> TrieNode for TrieNodeWithValue<T>
where
    T: Debug + Send + Sync + 'static,
{
    fn children(&self) -> &Children {
        &self.node.children
    }

    fn is_value_node(&self) -> bool {
        self.node.is_value_node
    }

    fn clone_node(&self) -> Box<dyn TrieNode> {
        Box::new(Self {
            node: self.node.clone(),
            value: Arc::clone(&self.value),
        })
    }

    fn children_mut(&mut self) -> &mut Children {
        &mut self.node.children
    }
}
