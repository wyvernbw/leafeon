use std::ops::{Add, Mul, Sub};

use ndarray::linalg::Dot;

#[cfg(feature = "serde")]
use serde::{Deserialize, Serialize};

use super::Array;

#[derive(Default, Debug, Clone)]
#[cfg_attr(feature = "serde", derive(Deserialize, Serialize))]
pub struct BaseOps;

impl<A> Add<Array<A, BaseOps>> for Array<A, BaseOps>
where
    A: Add<Output = A>,
{
    type Output = Array<A, BaseOps>;

    fn add(self, rhs: Array<A, BaseOps>) -> Self::Output {
        let data = self.data + rhs.data;
        Self::Output {
            data,
            operations: BaseOps,
        }
    }
}

impl<A> Sub<Array<A, BaseOps>> for Array<A, BaseOps>
where
    A: Sub<Output = A>,
{
    type Output = Array<A, BaseOps>;

    fn sub(self, rhs: Array<A, BaseOps>) -> Self::Output {
        let data = self.data - rhs.data;
        Self::Output {
            data,
            operations: BaseOps,
        }
    }
}

impl<A> Mul<Array<A, BaseOps>> for Array<A, BaseOps>
where
    A: Mul<Output = A>,
{
    type Output = Array<A, BaseOps>;

    fn mul(self, rhs: Array<A, BaseOps>) -> Self::Output {
        let data = self.data * rhs.data;
        Self::Output {
            data,
            operations: BaseOps,
        }
    }
}

impl<A> Dot<Array<A, BaseOps>> for Array<A, BaseOps>
where
    A: Dot<A>,
{
    type Output = <A as Dot<A>>::Output;

    fn dot(&self, rhs: &Array<A, BaseOps>) -> Self::Output {
        self.data.dot(&rhs.data)
    }
}

#[cfg(test)]
mod tests {
    use ndarray::array;
    use rstest::rstest;

    use super::*;
    use crate::array::{Array1, Array2};

    #[rstest]
    #[case(array![1, 2, 3], array![4, 5, 6])]
    fn test_add(
        #[case] a: impl Into<Array1<i32, BaseOps>>,
        #[case] b: impl Into<Array1<i32, BaseOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.clone() + b.data.clone();
        let result = a + b;
        assert_eq!(result.data, expected);
    }

    #[rstest]
    #[case(array![1, 2, 3], array![4, 5, 6])]
    #[should_panic]
    #[case(array![1, 2, 3], array![4, 5, 6, 7])]
    fn test_sub(
        #[case] a: impl Into<Array1<i32, BaseOps>>,
        #[case] b: impl Into<Array1<i32, BaseOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.clone() - b.data.clone();
        let result = a - b;
        assert_eq!(result.data, expected);
    }

    #[rstest]
    #[case(array![1, 2, 3], array![4, 5, 6])]
    #[should_panic]
    #[case(array![1, 2, 3], array![4, 5, 6, 7])]
    #[case(array![3, 10, 30], array![4, 5, 6])]
    fn test_mul(
        #[case] a: impl Into<Array1<i32, BaseOps>>,
        #[case] b: impl Into<Array1<i32, BaseOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.clone() * b.data.clone();
        let result = a * b;
        assert_eq!(result.data, expected);
    }

    #[rstest]
    #[case(array![1, 2, 3], array![4, 5, 6])]
    #[should_panic]
    #[case(array![1, 2, 3], array![4, 5, 6, 7])]
    fn test_dot(
        #[case] a: impl Into<Array1<i32, BaseOps>>,
        #[case] b: impl Into<Array1<i32, BaseOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.dot(&b.data);
        let result = a.dot(&b);
        assert_eq!(result, expected);
    }

    fn large_array(offset: i32) -> Array2<i32, BaseOps> {
        ndarray::Array2::from_shape_fn((256, 256), |(x, y)| x as i32 + y as i32 + offset).into()
    }

    #[rstest]
    #[should_panic]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5, 6], [7, 8, 9], [10, 11, 12]])]
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[4, 5], [6, 7], [8, 9]])] // Rectangular matrix multiplication
    #[case(array![[0, 0, 0], [0, 0, 0]], array![[0, 0], [0, 0], [0, 0]])] // All zero matrices
    #[case(array![[1]], array![[1]])] // Single-element (1x1) matrices
    #[case(array![[-1, -2, -3], [-4, -5, -6]], array![[1, 2, 3], [4, 5, 6], [7, 8, 9]])] // Negative values in 'a'
    #[case(array![[1, 2, 3], [4, 5, 6]], array![[-4, -5], [-6, -7], [-8, -9]])] // Negative values in 'b'
    #[case(array![[1, 2]], array![[3], [4]])] // Non-square matrices
    #[case(array![[1, 2, 3]], array![[4], [5], [6]])] // Row-vector times column-vector
    #[case(array![[1], [2], [3]], array![[4, 5, 6]])] // Column-vector times row-vector
    #[case(large_array(0), large_array(16))]
    fn test_dot_2(
        #[case] a: impl Into<Array2<i32, BaseOps>>,
        #[case] b: impl Into<Array2<i32, BaseOps>>,
    ) {
        let a = a.into();
        let b = b.into();
        let expected = a.data.dot(&b.data);
        let result = a.dot(&b);
        assert_eq!(result, expected);
    }
}
