import random

class snake:
  def __init__(self, _XSIZE, _YSIZE):
    self.XSIZE = _XSIZE
    self.YSIZE = _YSIZE
    self.reset()
    self.food = [0,2]

  def reset(self):

    self.snake = [[8, 10], [8, 9], [8, 8], [8, 7], [8, 6], [8, 5], [8, 4], [
        8, 3], [8, 2], [8, 1], [8, 0]]  # Initial snake co-ordinates [ypos,xpos]
    self.food = self.place_food()
    self.snake_direction = "right"

  def place_food(self):
    self.food = [random.randint(1, (self.YSIZE-2)),
                 random.randint(1, (self.XSIZE-2))]
    while (self.food in self.snake):
      self.food = [random.randint(
          1, (self.YSIZE-2)), random.randint(1, (self.XSIZE-2))]
    return(self.food)

  def update_snake_position(self):
    self.snake.insert(0, [self.snake[0][0] + (self.snake_direction == "down" and 1) + (self.snake_direction == "up" and -1),
                      self.snake[0][1] + (self.snake_direction == "left" and -1) + (self.snake_direction == "right" and 1)])

  def food_eaten(self):
    # When snake eats the food
    if self.snake[0] == self.food:
      return True
    else:
      # [1] If it does not eat the food, it moves forward and so last tail item is removed
      last = self.snake.pop()
      return False

  def snake_turns_into_self(self):
    if self.snake[0] in self.snake[1:]:
      return True
    else:
      return False

  def snake_hit_wall(self):
    if self.snake[0][0] == 0 or self.snake[0][0] == (self.YSIZE-1) or self.snake[0][1] == 0 or self.snake[0][1] == (self.XSIZE-1):
      return True
    else:
      return False

  def senseWall(self, pos):
    return(pos[0] <= 0 or pos[0] >= (self.YSIZE-1) or pos[1] <= 0 or pos[1] >= (self.XSIZE-1))

  def senseTail(self, pos):
    return pos in self.snake

  def leftWall(self):
    return self.senseWall([self.snake[0][0],self.snake[0][1]-1])

  def leftTail(self):
    return self.senseTail([self.snake[0][0],self.snake[0][1]-1])

  def leftFood(self):
    return self.food[1] < self.snake[0][1]

  def rightWall(self):
    return self.senseWall([self.snake[0][0],self.snake[0][1]+1])

  def rightTail(self):
    return self.senseTail([self.snake[0][0],self.snake[0][1]+1])

  def rightFood(self):
    return self.food[1] > self.snake[0][1]

  def upWall(self):
    return self.senseWall([self.snake[0][0]-1,self.snake[0][1]])

  def upTail(self):
    return self.senseTail([self.snake[0][0]-1,self.snake[0][1]])

  def upFood(self):
    return self.food[0] < self.snake[0][0]

  def downWall(self):
    return self.senseWall([self.snake[0][0]+1,self.snake[0][1]])

  def downTail(self):
    return self.senseTail([self.snake[0][0]+1,self.snake[0][1]])

  def downFood(self):
    return self.food[0] > self.snake[0][0]

  def getInputs(self):
      return [  self.leftWall(), self.leftTail(), self.leftFood(),
                self.rightWall(), self.rightTail(), self.rightFood(),
                self.upWall(), self.upTail(), self.upFood(),
                self.downWall(), self.downTail(), self.downFood(),
                ]