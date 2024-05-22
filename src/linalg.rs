use std::vec;

use crate::config::{Float, EPSILON, ToFloat};

pub fn float_eq(f1: Float, f2: Float) -> bool {
    f1 >= f2 - EPSILON && f1 <= f2 + EPSILON
}

pub fn float_min(f1: Float, f2: Float) -> Float {
    // !!!only works for positive float
    f32::min(f1, f2)
    // f1.partial_min(f2).unwrap()
}

pub fn float_max(f1: Float, f2: Float) -> Float {
    // !!!only works for positive float
    f32::max(f1, f2)
    // f1.partial_max(f2).unwrap()
}

pub fn log(x: Float) -> Float {
    x.ln()
}

pub struct Matrix<T: std::ops::Add + std::ops::DivAssign> {
    pub data: Vec<Vec<T>>,
}

impl<T> Matrix<T> 
where Vec<T>: FromIterator<<T as std::ops::Add>::Output> + FromIterator<<T as std::ops::Sub>::Output>
+ FromIterator<<T as std::ops::Div>::Output>, 
T: std::ops::Add + std::ops::DivAssign + Clone + std::ops::Sub + std::cmp::PartialEq + std::ops::Div
 + ToFloat + std::ops::Div<Output = T> + Copy
{
    pub fn new(data: &Vec<Vec<T>>) -> Matrix<T> {
        let data: Vec<Vec<T>> = data.iter().map(|inner_vec| inner_vec.clone()).collect();
        let m: Matrix<T> = Matrix { data };
        m
    }

    pub fn get_data(& self) -> &Vec<Vec<T>> {
        & self.data
    }

    pub fn get_nth_row(& self, n: usize) -> Vec<T> {
        self.data[n].clone()
    }

    pub fn get_nth_column(& self, n: usize) -> Vec<T> {
        self.data.iter().map(|x| *x.get(n).unwrap()).collect::<Vec<_>>()
    }

    pub fn shape(& self) -> (usize, usize) {
        let n_row = self.data.len();
        if n_row == 0 {
            return (0, 0)
        }
        let n_col = self.data.get(0).unwrap().len();
        (n_row, n_col)
    }

    pub fn add(& self, m: &Matrix<T>) -> Matrix<T> {
        let shape = self.shape();
        if shape.0 == 0 {
            let data = Vec::new();
            let t: Matrix<T> = Matrix { data };
            return t;
        }

        let n_row = shape.0;
        let n_col = shape.1;
        assert!(n_col > 0);

        let mut data = Vec::new();

        for row in 0..n_row {
            let v1 = self.get_nth_row(row);
            let v2 = m.get_nth_row(row);
            let v = v1.iter().zip(v2).map(|(&x, y)| x + y).collect();
            data.push(v);
        }

        let result: Matrix<T> = Matrix { data };
        result
    }

    pub fn subtract(& self, m: &Matrix<T>) -> Matrix<T> {
        let shape = self.shape();
        if shape.0 == 0 {
            let data = Vec::new();
            let t: Matrix<T> = Matrix { data };
            return t;
        }

        let n_row = shape.0;
        let n_col = shape.1;
        assert!(n_col > 0);

        let mut data = Vec::new();

        for row in 0..n_row {
            let v1 = self.get_nth_row(row);
            let v2 = m.get_nth_row(row);
            let v = v1.iter().zip(v2).map(|(&x, y)| x - y).collect();
            data.push(v);
        }

        let result: Matrix<T> = Matrix { data };
        result
    }

    pub fn divide(& self, m: &Matrix<T>) -> Matrix<Float> {
        let shape = self.shape();
        if shape.0 == 0 {
            let data = Vec::new();
            let t: Matrix<Float> = Matrix { data };
            return t;
        }

        let n_row = shape.0;
        let n_col = shape.1;
        assert!(n_col > 0);

        let mut data = Vec::new();

        for row in 0..n_row {
            let v1 = self.get_nth_row(row);
            let v2 = m.get_nth_row(row);

            let v: Vec<Float> = v1
            .iter()
            .zip(v2.iter())
            .map(|(&a, &b)| {
                if b.as_float() == (0.0 as Float) {
                    0.0 as Float
                } else {
                    let c: T = a / b;
                    c.as_float()
                }
            })
            .collect();

            data.push(v);
        }

        let result: Matrix<Float> = Matrix { data };
        result
    }

    pub fn divide_scalar(& self, scalar: T) -> Matrix<Float> {
        let shape = self.shape();
        if shape.0 == 0 {
            let data = Vec::new();
            let t: Matrix<Float> = Matrix { data };
            return t;
        }

        let n_row = shape.0;
        let n_col = shape.1;
        assert!(n_col > 0);

        let mut data = Vec::new();
        for _ in 0..n_row {
            data.push(vec![scalar; n_col])
        }

        let m: Matrix<T> = Matrix { data };
        self.divide(&m)
    }

    pub fn sum(& self, axis: usize) -> Vec<Float> {
        let shape = self.shape();
        let n_row = shape.0;
        let n_col = shape.1;

        if n_row == 0 {
            let data = Vec::new();
            return data;
        }

        assert!(n_col > 0);

        let mut result = Vec::new();

        for row in 0..n_row {
            let v = self.get_nth_row(row);
            let v: Vec<Float> = v.iter().map(|&item| item.as_float()).collect();

            if axis == 0 {
                if result.len() == 0 {
                    result.extend(v.clone());
                } else {
                    result = result.iter().zip(v).map(|(&x, y)| x + y).collect();
                }
            } else {
                let v = v.iter().sum();
                result.push(v);
            }
        }

        result
    }
}
