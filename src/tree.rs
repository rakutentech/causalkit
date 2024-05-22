use serde::{Serialize, Deserialize};

use crate::split_info::{SplitInfo, SplitInfoTrait};
use crate::config::Float;
use crate::linalg::Matrix;

pub trait Tree {
    type T: Clone + SplitInfoTrait;

    fn new() -> Self;
    fn get_level(&mut self) -> &mut Vec<usize>;
    fn get_left_children(&mut self) -> &mut Vec<Option<usize>>;
    fn get_right_children(&mut self) -> &mut Vec<Option<usize>>;
    fn get_split(&mut self) -> &mut Vec<Option<Self::T>>;

    fn add_split(& mut self, node: usize, split_info: Self::T) {
        if let Some(x) = self.get_split().get_mut(node) {
            *x = Some(split_info);
        }
    }

    fn get_split_info(&mut self, node: usize) -> Option<Self::T> {
        let split_info = self.get_split().get(node).unwrap();
        match split_info {
            Some(value) => Some(value.clone()),
            None => None
        }
    }

    fn add_left(& mut self, node: usize) -> usize {
        let left = self.get_level().len();

        if let Some(x) = self.get_left_children().get_mut(node) {
            *x = Some(left);
        }

        let tree_level = self.get_level();
        let level = tree_level.get(node).unwrap();
        tree_level.push(level + 1);

        self.get_left_children().push(None);
        self.get_right_children().push(None);
        self.get_split().push(None);

        left
    }

    fn add_right(& mut self, node: usize) -> usize {
        let right = self.get_level().len();

        if let Some(x) = self.get_right_children().get_mut(node) {
            *x = Some(right);
        }

        let tree_level = self.get_level();
        let level = tree_level.get(node).unwrap();
        tree_level.push(level + 1);

        self.get_left_children().push(None);
        self.get_right_children().push(None);
        self.get_split().push(None);

        right
    }

    fn is_leaf(&mut self, n: usize) -> bool {
        self.get_left_children().get(n).unwrap().is_none()
    }

    fn depth(&mut self, n: usize) -> usize {
        *self.get_level().get(n).unwrap()
    }

    fn size(&mut self) -> usize {
        self.get_level().len()
    }

    fn from_string(s: &String) -> Self;

    fn to_string(& self) -> String;

    fn get_uplift(&mut self, node: usize) -> Vec<Float> {
        let split_info = self.get_split_info(node).unwrap();
        let summary = split_info.get_summary();
        let matrix_summary: Matrix<Float> = Matrix::new(& summary);
        let prob = matrix_summary.get_nth_column(0);
        let n = prob.len();

        let mut uplift = Vec::new();
        for i in 1..n {
            uplift.push(prob.get(i).unwrap() - prob.get(0).unwrap());
        }

        uplift
    }
}

#[derive(Deserialize, Serialize, Clone)]
pub struct ClassificationTree {
    pub level: Vec<usize>,
    pub left_children: Vec<Option<usize>>,
    pub right_children: Vec<Option<usize>>,
    pub split: Vec<Option<SplitInfo>>
}

impl Tree for ClassificationTree {
    type T = SplitInfo;

    fn new() -> ClassificationTree {
        let level = vec![0; 1];
        let mut left_children = Vec::new();
        left_children.push(None);

        let mut right_children = Vec::new();
        right_children.push(None);

        let mut split = Vec::new();
        split.push(None);

        ClassificationTree {
            level,
            left_children,
            right_children,
            split
        }
    }

    fn from_string(s: &String) -> Self {
        serde_json::from_str(s).unwrap()
    }

    fn to_string(& self) -> String {
        let json_string = serde_json::to_string(self).unwrap();
        json_string
    }

    fn get_level(&mut self) -> &mut Vec<usize> {
        &mut self.level
    }

    fn get_left_children(&mut self) -> &mut Vec<Option<usize>> {
        &mut self.left_children
    }
    fn get_right_children(&mut self) -> &mut Vec<Option<usize>> {
        & mut self.right_children
    }

    fn get_split(&mut self) -> &mut Vec<Option<Self::T>> {
        & mut self.split
    }

}

#[derive(Deserialize, Serialize, Clone)]
pub struct RegressionTree {
    pub level: Vec<usize>,
    pub left_children: Vec<Option<usize>>,
    pub right_children: Vec<Option<usize>>,
    pub split: Vec<Option<SplitInfo>>
}

impl Tree for RegressionTree {
    type T = SplitInfo;

    fn new() -> RegressionTree {
        let level = vec![0; 1];
        let mut left_children = Vec::new();
        left_children.push(None);

        let mut right_children = Vec::new();
        right_children.push(None);

        let mut split = Vec::new();
        split.push(None);

        RegressionTree {
            level,
            left_children,
            right_children,
            split
        }
    }

    fn from_string(s: &String) -> Self {
        serde_json::from_str(s).unwrap()
    }

    fn to_string(& self) -> String {
        let json_string = serde_json::to_string(self).unwrap();
        json_string
    }

    fn get_level(&mut self) -> &mut Vec<usize> {
        &mut self.level
    }

    fn get_left_children(&mut self) -> &mut Vec<Option<usize>> {
        &mut self.left_children
    }
    fn get_right_children(&mut self) -> &mut Vec<Option<usize>> {
        & mut self.right_children
    }

    fn get_split(&mut self) -> &mut Vec<Option<Self::T>> {
        & mut self.split
    }
}
