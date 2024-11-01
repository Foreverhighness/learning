use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

type Children = HashMap<char, Arc<dyn TrieNode>>;

pub trait TrieNode: Debug + Send + Sync + Any {
    fn children(&self) -> &Children;
    fn children_mut(&mut self) -> &mut Children;
    fn is_value_node(&self) -> bool;
    fn clone_node(&self) -> Box<dyn TrieNode>;
}

#[derive(Clone, Default)]
pub struct NodeWithoutValue {
    children: Children,
    is_value_node: bool,
}

impl Debug for NodeWithoutValue {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "")
    }
}

impl NodeWithoutValue {
    pub fn new_box() -> Box<dyn TrieNode> {
        Box::<NodeWithoutValue>::default()
    }

    pub fn with_children(children: Children) -> Box<dyn TrieNode> {
        Box::new(Self {
            children,
            is_value_node: false,
        })
    }
}

impl TrieNode for NodeWithoutValue {
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
pub struct NodeWithValue<T> {
    node: NodeWithoutValue,
    pub value: Arc<T>,
}

impl<T: Debug> Debug for NodeWithValue<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.value)
    }
}

pub trait NodeValueMarkerTrait: Debug + Send + Sync + 'static {}
impl<T> NodeValueMarkerTrait for T where T: Debug + Send + Sync + 'static {}

impl<T> NodeWithValue<T>
where
    T: NodeValueMarkerTrait,
{
    pub fn new_box(value: Arc<T>) -> Box<dyn TrieNode> {
        Self::with_children(HashMap::default(), value)
    }

    pub fn with_children(children: Children, value: Arc<T>) -> Box<dyn TrieNode> {
        let node = NodeWithoutValue {
            is_value_node: true,
            children,
        };
        Box::new(Self { node, value })
    }
}

impl<T: NodeValueMarkerTrait> TrieNode for NodeWithValue<T> {
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
