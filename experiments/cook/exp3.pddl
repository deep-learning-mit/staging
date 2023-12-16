(define (domain bagel-preparation)
  (:requirements :strips :typing)
  (:types
    food-container - object
    food - object
    utensil - object
    appliance - object
  )

  (:predicates
    (is-cut ?f - food)
    (is-cooked ?f - food)
    (is-on ?o - object ?s - appliance)
    (is-clean ?u - utensil)
    (is-hot ?a - appliance)
    (is-plugged-in ?a - appliance)
  )

  (:action cut-bagel
    :parameters (?b - food ?k - utensil ?p - food-container)
    :precondition (and (not (is-cut ?b)) (is-clean ?k) (is-on ?b ?p))
    :effect (is-cut ?b)
  )

  (:action cook-bagel
    :parameters (?b - food ?p - appliance)
    :precondition (and (is-cut ?b) (is-hot ?p) (is-plugged-in ?p) (not (is-cooked ?b)))
    :effect (is-cooked ?b)
  )

  (:action heat-pan
    :parameters (?p - appliance)
    :precondition (and (not (is-hot ?p)) (is-plugged-in ?p))
    :effect (is-hot ?p)
  )

  (:action plug-in-appliance
    :parameters (?a - appliance)
    :precondition (not (is-plugged-in ?a))
    :effect (is-plugged-in ?a)
  )

  (:action place-object
    :parameters (?o - object ?a - appliance)
    :precondition (not (is-on ?o ?a))
    :effect (is-on ?o ?a)
  )
)