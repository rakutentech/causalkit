use crate::dmatrix::DMatrix;
use crate::config::{Int, Float};

pub struct Count {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Vec<Float>>>,
}

impl Count {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> Count {

        let feature = m.feature.get(feature_id).unwrap();
        let treatment = m.treatments.get(treatment_id).unwrap();
        let y = & m.response;
        let weight = & m.weights;
        let first_dim = 2;
        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let stat = match iscat {
            x if x => Count::cat(feature, treatment, y, weight, indices,
                first_dim, second_dim, third_dim),
            _ => Count::cont(feature, treatment, y, weight, indices,
                first_dim, second_dim, third_dim)
        };

        Count {feature_id, iscat, treatment_id, stat}
    }

    fn cat(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], first_dim: usize, second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Vec<Float>>> {
        
        let mut stat: Vec<Vec<Vec<Float>>> = vec![vec![vec![0.0; third_dim]; second_dim]; first_dim];

        for v in indices.iter() {
            let f: usize = match feature[*v] {
                Some(p) => p as usize,
                None => third_dim - 1
            };

            let t = treatment[*v] as usize;
            let r = y[*v] as usize;
            let s = weight[*v];
            stat[r][t][f] = stat[r][t][f] + s;
        }

        stat
    }

    fn cont(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], first_dim: usize, second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Vec<Float>>> {
     
        let stat = Count::cat(feature, treatment, y, weight, indices,
            first_dim, second_dim, third_dim);

        let mut acc: Vec<Vec<Vec<Float>>> = vec![vec![vec![0.0; third_dim]; second_dim]; first_dim];
        for (idx_y, arr2) in stat.iter().enumerate() {
            for (idx_t, arr1) in arr2.iter().enumerate() {
                let accumulated_sum: Vec<Float> = arr1.iter().scan(0.0, |sum, &x| {
                    *sum += x;
                    Some(*sum)
                }).collect();

                acc[idx_y][idx_t] = accumulated_sum;
            }
        }

        acc
    }
}

pub struct Sum {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Float>>,
}

impl Sum {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> Sum {

        let feature = m.feature.get(feature_id).unwrap();
        let treatment = m.treatments.get(treatment_id).unwrap();
        let y = & m.response;
        let weight = & m.weights;
        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let stat = match iscat {
            x if x => Sum::cat(feature, treatment, y, weight, indices,
                second_dim, third_dim),
            _ => Sum::cont(feature, treatment, y, weight, indices,
                second_dim, third_dim)
        };

        Sum {feature_id, iscat, treatment_id, stat}
    }

    fn cat(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
        
        let mut stat: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];

        for v in indices.iter() {
            let f: usize = match feature[*v] {
                Some(p) => p as usize,
                None => third_dim - 1
            };

            let t = treatment[*v] as usize;
            stat[t][f] = stat[t][f] + y[*v] * weight[*v];
        }

        stat
    }

    fn cont(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
     
        let stat = Sum::cat(feature, treatment, y, weight, indices,
            second_dim, third_dim);

        let mut acc: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];
        for (idx_t, arr1) in stat.iter().enumerate() {
            let accumulated_sum: Vec<Float> = arr1.iter().scan(0.0, |sum, &x| {
                *sum += x;
                Some(*sum)
            }).collect();

            acc[idx_t] = accumulated_sum;
        }

        acc
    }
}

pub struct CountNoY {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Float>>,
}

impl CountNoY {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> CountNoY {

        let feature = m.feature.get(feature_id).unwrap();
        let treatment = m.treatments.get(treatment_id).unwrap();
        let weight = & m.weights;
        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let stat = match iscat {
            x if x => CountNoY::cat(feature, treatment, weight, indices,
                second_dim, third_dim),
            _ => CountNoY::cont(feature, treatment, weight, indices,
                second_dim, third_dim)
        };

        CountNoY {feature_id, iscat, treatment_id, stat}
    }

    fn cat(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
        
        let mut stat: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];

        for v in indices.iter() {
            let f: usize = match feature[*v] {
                Some(p) => p as usize,
                None => third_dim - 1
            };

            let t = treatment[*v] as usize;
            stat[t][f] = stat[t][f] + weight[*v];
        }

        stat
    }

    fn cont(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
     
        let stat = CountNoY::cat(feature, treatment, weight, indices,
            second_dim, third_dim);

        let mut acc: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];
        for (idx_t, arr1) in stat.iter().enumerate() {
            let accumulated_sum: Vec<Float> = arr1.iter().scan(0.0, |sum, &x| {
                *sum += x;
                Some(*sum)
            }).collect();

            acc[idx_t] = accumulated_sum;
        }

        acc
    }
}

pub struct Mean {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Float>>,
}

impl Mean {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> Mean {

        let sum = Sum::calculate(m, feature_id, iscat, treatment_id, indices, 
            feature_size, treatment_size).stat;

        let cnt = CountNoY::calculate(m, feature_id, iscat, treatment_id, indices, 
            feature_size, treatment_size).stat;

        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let mut stat: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];

        for idx_t in 0..second_dim {
            let u = sum.get(idx_t).unwrap();
            let w = cnt.get(idx_t).unwrap();
            let v: Vec<Float> = u.iter().zip(w.iter()).map(|(&u_val, &w_val)| u_val / w_val).collect();
            stat[idx_t] = v;
        }

        Mean { feature_id, iscat, treatment_id, stat }
    }
}

pub struct SecondOrderSum {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Float>>,
}

impl SecondOrderSum {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> SecondOrderSum {

        let feature = m.feature.get(feature_id).unwrap();
        let treatment = m.treatments.get(treatment_id).unwrap();
        let y = & m.response;
        let weight = & m.weights;
        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let stat = match iscat {
            x if x => SecondOrderSum::cat(feature, treatment, y, weight, indices,
                second_dim, third_dim),
            _ => SecondOrderSum::cont(feature, treatment, y, weight, indices,
                second_dim, third_dim)
        };

        SecondOrderSum {feature_id, iscat, treatment_id, stat}
    }

    fn cat(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
        
        let mut stat: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];

        for v in indices.iter() {
            let f: usize = match feature[*v] {
                Some(p) => p as usize,
                None => third_dim - 1
            };

            let t = treatment[*v] as usize;
            stat[t][f] = stat[t][f] + y[*v].powi(2) * weight[*v];
        }

        stat
    }

    fn cont(feature: &Vec<Option<Int>>, treatment: &Vec<Int>, y: &Vec<Float>, 
        weight: &Vec<Float>, indices: &[usize], second_dim: usize, 
        third_dim: usize) -> Vec<Vec<Float>> {
     
        let stat = SecondOrderSum::cat(feature, treatment, y, weight, indices,
            second_dim, third_dim);

        let mut acc: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];
        for (idx_t, arr1) in stat.iter().enumerate() {
            let accumulated_sum: Vec<Float> = arr1.iter().scan(0.0, |sum, &x| {
                *sum += x;
                Some(*sum)
            }).collect();

            acc[idx_t] = accumulated_sum;
        }

        acc
    }
}

pub struct SecondMoment {
    pub feature_id: usize,
    pub iscat: bool,
    pub treatment_id: usize,
    pub stat: Vec<Vec<Float>>,
}

impl SecondMoment {

    pub fn calculate(m: &DMatrix, feature_id: usize, iscat: bool, treatment_id: usize, 
        indices: &[usize], feature_size: usize, treatment_size: usize) -> SecondMoment {

        let second_dim = treatment_size;
        let third_dim = feature_size + 1;

        let sum = SecondOrderSum::calculate(m, feature_id, iscat, treatment_id, indices, 
            feature_size, treatment_size).stat;

        let cnt = CountNoY::calculate(m, feature_id, iscat, treatment_id, indices, 
            feature_size, treatment_size).stat;

        let mut stat: Vec<Vec<Float>> = vec![vec![0.0; third_dim]; second_dim];

        for idx_t in 0..second_dim {
            let u = sum.get(idx_t).unwrap();
            let w = cnt.get(idx_t).unwrap();
            let v: Vec<Float> = u.iter().zip(w.iter()).map(|(&u_val, &w_val)| u_val / w_val).collect();
            stat[idx_t] = v;
        }

        SecondMoment { feature_id, iscat, treatment_id, stat }
    }
}
