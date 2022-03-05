import turtle
import time
import numpy as np

def run_game(network, display, snake_game, headless):
  score = 0
  timeAlive = 0
  timeSinceEaten = 0
  snake_game.reset()
  if not headless:
    display.reset(snake_game.snake)
    display.win.update()
  snake_game.place_food()
  game_over = False

  while not game_over:
    timeAlive+=1
    timeSinceEaten+=1
    # ****YOUR AI BELOW HERE******************

    output = network.feedForward(snake_game.getInputs())
    decision = np.argmax(output, axis=0)

    if decision == 0:
      snake_game.snake_direction = "left"
    elif decision == 1:
      snake_game.snake_direction = "right"
    elif decision == 2:
      snake_game.snake_direction = "up"
    elif decision == 3:
      snake_game.snake_direction = "down"

    # ****YOUR AI ABOVE HERE******************

    snake_game.update_snake_position()

    # Check if food is eaten
    if snake_game.food_eaten():
      if (len(snake_game.snake) != (snake_game.XSIZE-2) * (snake_game.YSIZE-2)):
        snake_game.place_food()
        timeSinceEaten = 0
        score += 1
        if not headless:
          display.add_snake_segment()
      else:
        game_over = True
    # Game over if the snake runs over itself
    if snake_game.snake_turns_into_self():
      game_over = True
      #print("Snake turned into itself!")

    # Game over if the snake goes through a wall
    if snake_game.snake_hit_wall():
      game_over = True
      #print("Snake hit a wall!")

    if(min((5*len(snake_game.snake),200)) < timeSinceEaten):
      timeAlive-=timeSinceEaten
      game_over = True

    if not headless:
      display.update_food(snake_game.food)
      display.update_segment_positions(snake_game.snake)
      display.win.update()
      # Change this to modify the speed the game runs at when displayed.
      time.sleep(0.02)

  if not headless:
    turtle.done()
    try:
      turtle.bye()
    except turtle.Terminator:
      pass
    print("\nFINAL score - " + str(score))

  return (score ** 3) * timeAlive, score