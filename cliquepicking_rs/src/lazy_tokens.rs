#[derive(Debug)]
pub struct LazyTokens {
    tokens: Vec<usize>,
    depth: usize,
    changed: Vec<Vec<(usize, usize)>>,
}

impl LazyTokens {
    pub fn new(n: usize) -> LazyTokens {
        LazyTokens {
            tokens: vec![0; n],
            depth: 0,
            changed: Vec::new(),
        }
    }

    pub fn set(&mut self, i: usize) {
        if self.tokens[i] == self.depth {
            return;
        }
        self.changed.last_mut().unwrap().push((i, self.tokens[i]));
        self.tokens[i] = self.depth;
    }

    pub fn check(&self, i: usize) -> bool {
        self.tokens[i] == self.depth
    }

    pub fn prepare(&mut self) {
        self.depth += 1;
        self.changed.push(Vec::new());
    }

    pub fn restore(&mut self) {
        for &(i, prev_value) in self.changed.last().unwrap().iter() {
            self.tokens[i] = prev_value;
        }
        self.depth -= 1;
        self.changed.pop();
    }
}
