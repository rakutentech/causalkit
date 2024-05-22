use crate::dmatrix::DMatrix;
use crate::split_info::SplitInfoTrait;
use crate::config::TreeConfig;

pub trait Strategy {
    type T: Clone + SplitInfoTrait;

    fn new(conf: &TreeConfig) -> Self;

    fn find_best_split(& self, m: &DMatrix, node_id: usize, feature_id: usize, 
        indices: &mut [usize]) -> Self::T;
}
