(define (domain kitchen-tasks)
  (:requirements :strips :typing :fluents)
  (:types
    food - object
    container - object
    utensil - object
    appliance - object
    cleaning-agent - object
    cleaning-tool - object
    surface - object
    - location
  ) 
  
  (:predicates
    (is-clean ?s - surface)
    (is-dirty ?s - surface)
    (is-cooked ?f - food)
    (is-raw ?f - food)
    (on ?x - object ?y - surface)
    (inside ?x - object ?y - container)
    (empty ?c - container)
    (contains ?c - container ?f - food)
    (holding ?u - utensil ?f - food)
    (is-hot ?a - appliance)
    (is-off ?a - appliance)
  )
  
  (:action cook
    :parameters (?f - food ?a - appliance)
    :precondition (and (is-raw ?f) (is-hot ?a))
    :effect (and (is-cooked ?f) (not (is-raw ?f)))
  )

  (:action clean
    :parameters (?s - surface ?c - cleaning-agent ?t - cleaning-tool)
    :precondition (is-dirty ?s)
    :effect (and (is-clean ?s) (not (is-dirty ?s)))
  )

  (:action prepare-sandwich
    :parameters (?b - food ?p - container)
    :precondition (and (contains ?p ?b) (is-clean ?p))
    :effect (and (inside ?b ?p) (not (empty ?p)))
  )
  
  ; ... Additional actions and objects based on the kitchen scene
)
