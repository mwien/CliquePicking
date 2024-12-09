#[derive(Clone, Debug, Eq, Hash, PartialEq)]
pub struct IndexSet(Vec<usize>);

// assume no duplicate elements
impl IndexSet {
    pub fn new() -> IndexSet {
        IndexSet(Vec::new())
    }
    pub fn from(mut set: Vec<usize>) -> IndexSet {
        set.sort();
        IndexSet(set)
    }

    pub fn from_sorted(set: Vec<usize>) -> IndexSet {
        IndexSet(set)
    }

    pub fn iter(&self) -> std::slice::Iter<'_, usize> {
        self.0.iter()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    pub fn first(&self) -> Option<usize> {
        self.iter().copied().next()
    }

    pub fn get(&self, pos: usize) -> usize {
        self.0[pos]
    }

    pub fn contains(&self, x: usize) -> bool {
        self.0.binary_search(&x).is_ok()
    }

    pub fn is_subset(&self, other: &IndexSet) -> bool {
        if self.len() > other.len() {
            return false;
        }
        let mut it = other.iter();
        for &el in self {
            loop {
                match it.next() {
                    Some(&x) => {
                        if x > el {
                            return false;
                        } else if x == el {
                            break;
                        }
                    }
                    None => return false,
                };
            }
        }
        true
    }

    pub fn intersection(&self, other: &IndexSet) -> IndexSet {
        let mut intersection_vec = Vec::new();
        let mut it = other.iter().peekable();
        for &el in self {
            while let Some(&&x) = it.peek() {
                if x < el {
                    it.next();
                } else if x == el {
                    intersection_vec.push(el);
                    it.next();
                } else {
                    break;
                }
            }
        }
        IndexSet::from_sorted(intersection_vec)
    }

    pub fn union(&self, other: &IndexSet) -> IndexSet {
        let mut union_vec = Vec::new();
        let mut it = other.iter().peekable();
        for &el in self {
            while let Some(&&x) = it.peek() {
                if x < el {
                    union_vec.push(x);
                    it.next();
                } else if x == el {
                    it.next();
                } else {
                    break;
                }
            }
            union_vec.push(el);
        }
        for &el in it {
            union_vec.push(el);
        }
        IndexSet::from_sorted(union_vec)
    }

    pub fn set_difference(&self, other: &IndexSet) -> IndexSet {
        let mut set_difference_vec = Vec::new();
        let mut it = other.iter().peekable();
        for &el in self {
            while let Some(&&x) = it.peek() {
                if x < el {
                    it.next();
                } else if x == el {
                    break;
                } else {
                    set_difference_vec.push(el);
                    break;
                }
            }
            if it.peek().is_none() {
                set_difference_vec.push(el); // do tests
            }
        }
        IndexSet::from_sorted(set_difference_vec)
    }

    pub fn equal_to_vec(&self, vec: &[usize]) -> bool {
        if self.len() != vec.len() {
            return false;
        }
        for &el in vec {
            if !self.contains(el) {
                return false;
            }
        }
        true
    }
    pub fn to_vec(&self) -> Vec<usize> {
        self.0.clone()
    }
}

impl IntoIterator for IndexSet {
    type Item = usize;
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.into_iter()
    }
}

impl<'a> IntoIterator for &'a IndexSet {
    type Item = &'a usize;
    type IntoIter = std::slice::Iter<'a, usize>;

    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}

impl Default for IndexSet {
    fn default() -> Self {
        IndexSet::new()
    }
}

#[cfg(test)]
mod tests {
    use super::IndexSet;

    #[test]
    fn from_basic_check() {
        let v: Vec<usize> = vec![5, 3, 4, 1, 2];
        let w: Vec<usize> = vec![1, 2, 3, 4, 5];
        let v_is = IndexSet::from(v);
        assert_eq!(
            v_is.into_iter().collect::<Vec<usize>>(),
            vec![1, 2, 3, 4, 5]
        );
        let w_is = IndexSet::from_sorted(w);
        assert_eq!(
            w_is.into_iter().collect::<Vec<usize>>(),
            vec![1, 2, 3, 4, 5]
        );
    }

    #[test]
    fn is_subset_basic_check() {
        let v_is = IndexSet::from(vec![6, 3, 1]);
        let w1_is = IndexSet::from(vec![5, 6, 3, 4]);
        let w2_is = IndexSet::from(vec![5, 6, 3, 4, 1]);
        assert!(!v_is.is_subset(&w1_is));
        assert!(v_is.is_subset(&w2_is));
    }

    #[test]
    fn intersection_basic_check() {
        let v_is = IndexSet::from(vec![1, 7, 3, 4, 2]);
        let w_is = IndexSet::from(vec![4, 6, 2]);
        assert_eq!(
            v_is.intersection(&w_is).into_iter().collect::<Vec<usize>>(),
            vec![2, 4]
        )
    }

    #[test]
    fn union_basic_check() {
        let v_is = IndexSet::from(vec![1, 7, 3, 4]);
        let w_is = IndexSet::from(vec![4, 6, 2]);
        assert_eq!(
            v_is.union(&w_is).into_iter().collect::<Vec<usize>>(),
            vec![1, 2, 3, 4, 6, 7]
        )
    }
    // TODO: add test for len and equal_to_vec
}
