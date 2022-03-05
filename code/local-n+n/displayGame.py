import turtle

class DisplayGame:
  def __init__( self, XSIZE, YSIZE):
    # SCREEN
    self.win = turtle.Screen()
    self.win.title("EVCO Snake game")
    self.win.bgcolor("grey")
    self.win.setup(width=(XSIZE*20)+40, height=(YSIZE*20)+40)
    # self.win.screensize((XSIZE*20)+20,(YSIZE*20)+20)
    self.win.tracer(0)

    # Snake Head
    self.head = turtle.Turtle()
    self.head.shape("square")
    self.head.color("black")

    # Snake food
    self.food = turtle.Turtle()
    self.food.shape("circle")
    self.food.color("yellow")
    self.food.penup()
    self.food.shapesize(0.55, 0.55)
    self.segments = []

  def reset(self, snake):
    self.segments = []
    self.head.penup()
    self.food.goto(-500, -500)
    self.head.goto(-500, -500)
    for i in range(len(snake)-1):
      self.add_snake_segment()
    self.update_segment_positions(snake)

  def update_food(self, new_food):
    self.food.goto(((new_food[1]-9)*20)+20, (((9-new_food[0])*20)-10)-20)

  def update_segment_positions(self, snake):
    self.head.goto(((snake[0][1]-9)*20)+20, (((9-snake[0][0])*20)-10)-20)
    for i in range(len(self.segments)):
      self.segments[i].goto(((snake[i+1][1]-9)*20)+20,
                            (((9-snake[i+1][0])*20)-10)-20)

  def add_snake_segment(self):
    self.new_segment = turtle.Turtle()
    self.new_segment.speed(0)
    self.new_segment.shape("square")
    # random.choice(["green", 'black', 'red', 'blue']))
    self.new_segment.color("green")
    self.new_segment.penup()
    self.segments.append(self.new_segment)