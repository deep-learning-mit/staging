(define (domain kitchen-task)
  (:requirements :strips :typing)
  (:types 
    item - object
    food - item
    utensil - item
    container - item
  )

  (:predicates
    (is_clean ?item - item)
    (is_cut ?food - food)
    (is_cooked ?food - food)
    (is_on ?item - item ?container - container)
    (is_empty ?container - container)
  )

  (:action pick_up
    :parameters (?item - item ?container - container)
    :precondition (and (is_on ?item ?container))
    :effect (and (not (is_on ?item ?container)))
  )

  (:action put_down
    :parameters (?item - item ?container - container)
    :precondition (not (is_on ?item ?container))
    :effect (and (is_on ?item ?container))
  )

  (:action cut
    :parameters (?food - food)
    :precondition (and (not (is_cut ?food)))
    :effect (and (is_cut ?food))
  )

  (:action cook
    :parameters (?food - food)
    :precondition (and (not (is_cooked ?food)) (is_cut ?food))
    :effect (and (is_cooked ?food))
  )
)
