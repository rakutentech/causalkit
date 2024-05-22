use std::vec;
use std::default::Default;

#[derive(Default, Clone, Debug)]
pub struct Partition {
    pub data_ids: Vec<usize>,
    pub start: Vec<usize>,
    pub size: Vec<usize>,
}

impl Partition {

    pub fn new() -> Partition {
        Partition::default()
    }

    pub fn subsample(indices: Vec<usize>) -> Partition {
        let n = indices.len();

        let data_ids = indices;
        let start = vec![0; 1];
        let size = vec![n; 1];
        Partition {
            data_ids,
            start,
            size
        }
    }

    pub fn get_indices(&mut self, node_id: &usize) -> &mut [usize] {
        let start = self.start.get(*node_id).unwrap();
        let end = start + self.size.get(*node_id).unwrap();

        &mut self.data_ids[*start..end]
    }

    pub fn split(&mut self, node: usize, left_size: usize) {
        let start = self.start.get(node).unwrap().clone();
        let size = self.size.get(node).unwrap().clone();

        self.start.push(start);
        self.size.push(left_size);
        self.start.push(start + left_size);
        self.size.push(size - left_size);
    }

    pub fn refresh(total: usize, n_nodes: usize) -> Partition {
        let data_ids: Vec<usize> = (0..total).collect();
        let mut start = vec![0; 1];
        let mut size = vec![total; 1];

        for _ in 1..n_nodes {
            start.push(0);
            size.push(0);
        }

        Partition {
            data_ids,
            start,
            size
        }
    }

    pub fn set_node_range(&mut self, n: usize, start: usize, size: usize) {
        if let Some(x) = self.start.get_mut(n) {
            *x = start;
        }

        if let Some(x) = self.size.get_mut(n) {
            *x = size;
        }
    }
}
