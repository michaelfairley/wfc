use std::collections::VecDeque;

use rand::Rng;

#[derive(Copy,Clone)]
pub enum Rotation {
  None,
  R90,
  R180,
  R270,
}

// enum Symmetry {

// }

struct Wave {
  width: usize,
  height: usize,
  num_items: usize,
  possibilities: Vec<bool>,
}

impl Wave {
  fn index(&self, x: usize, y: usize, choice: usize) -> usize {
    (y * self.width + x) * self.num_items + choice
  }

  fn get(&self, x: usize, y: usize, choice: usize) -> bool {
    self.possibilities[self.index(x, y, choice)]
  }

  fn unset(&mut self, x: usize, y: usize, choice: usize) {
    let index = self.index(x, y, choice);
    self.possibilities[index] = false;
  }

  fn choices<'a>(&'a mut self, x: usize, y: usize) -> impl Iterator<Item=usize> + 'a {
    (0..self.num_items).filter(|&c| self.get(x, y, c))
  }

  fn num_possibilities(&mut self, x: usize, y: usize) -> usize {
    (0..self.num_items).filter(|&c| self.get(x, y, c)).count()
  }

  fn find_cell_with_fewest_choices(&self) -> ((usize, usize), Vec<usize>) {
    let mut lowest = 0;
    let mut lowest_i = (0, 0);

    for x in 0..self.width {
      for y in 0..self.height {
        let n = self.num_possibilities(x, y);
        if n == 0 { panic!("Optionless cell"); }

        if lowest == 0 || (n != 1 && n < lowest) {
          lowest = n;
          lowest_i = (x, y);
        }
      }
    }

    (lowest_i, self.choices(lowest_i.0, lowest_i.1).collect())
  }

  fn find_cell_with_fewest_choices(&self, (x, y): (usize, usize), choice: usize) {
    // THIS IS WHERE YOU LEFT OFF
  }
}

pub fn generate(
  (width, height): (usize, usize),
  num_items: usize,
) -> Vec<usize> {
  let mut rng = rand::thread_rng();

  let mut wave = Wave{
    width,
    height,
    num_items,
    possibilities: vec![true; width * height * num_items],
  };

  loop {
    let (next_to_resolve, choices) = wave.find_cell_with_fewest_choices();

    if choices.len() == 1 {
      break;
    } else if choices.len() == 0 {
      panic!("Ran out of chocies");
    }

    let choice = rng.choose(&choices);

    wave.unset_all_but(next_to_resolve, choice);

  //   let mut queue = VecDeque::new();
  //   queue.push_front(next_to_resolve);

  //   while let Some(next) = queue.pop_front() {
  //     let ann = acceptable_north_neighbors(next);
  //     let changed = wave.unset_inverse(next_to_resolve + north, ann);
  //     if change { add to queue }
  //     // + other dirs
  //   }

  }

  (0..width*height).map(|_| {
    rng.gen_range(0, num_items)
  }).collect()
}
