(define (domain kitchen_tasks)
  (:requirements :strips :typing)

  (:types 
    food - object
    tool - object
    container - object
    surface - object
  )

  (:predicates
    (isCut ?f - food)
    (isCooked ?f - food)
    (isClean ?t - tool)
    (isDirty ?t - tool)
    (on ?f - food ?s - surface)
    (has ?c - container ?f - food)
    (empty ?c - container)
    (stoveOff)
    (stoveOn)
  )

  (:action cut_food
    :parameters (?f - food ?t - tool ?s - surface)
    :precondition (and (not (isCut ?f)) (on ?f ?s))
    :effect (and (isCut ?f))
  )

  (:action cook_food
    :parameters (?f - food ?s - surface)
    :precondition (and (isCut ?f) (stoveOff))
    :effect (and (stoveOn) (isCooked ?f))
  )

  (:action clean_tool
    :parameters (?t - tool)
    :precondition (isDirty ?t)
    :effect (and (isClean ?t) (not (isDirty ?t)))
  )

  (:action turn_stove_on
    :precondition (stoveOff)
    :effect (and (stoveOn) (not (stoveOff)))
  )

  (:action turn_stove_off
    :precondition (stoveOn)
    :effect (and (not (stoveOn)) (stoveOff))
  )

  (:action place_food_on_surface
    :parameters (?f - food ?s - surface)
    :precondition (empty ?s)
    :effect (on ?f ?s)
  )

  (:action remove_food_from_surface
    :parameters (?f - food ?s - surface)
    :precondition (on ?f ?s)
    :effect (empty ?s)
  )

  (:action dirty_tool
    :parameters (?t - tool)
    :precondition (isClean ?t)
    :effect (and (isDirty ?t) (not (isClean ?t)))
  )
)
