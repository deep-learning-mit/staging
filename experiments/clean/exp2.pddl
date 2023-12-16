(define (domain kitchen_tasks)
  (:requirements :strips :typing)
  (:types location - object
          food - object
          container - object
          utensil - object
          cleaning_supplies - object)

  (:predicates
    (at ?o - object ?l - location)
    (clean ?o - object)
    (dirty ?o - object)
    (cut ?f - food)
    (cooked ?f - food)
    (empty ?c - container)
    (contains ?c - container ?o - object)
  )

  (:action move
    :parameters (?o - object ?from - location ?to - location)
    :precondition (at ?o ?from)
    :effect (and (at ?o ?to) (not (at ?o ?from)))
  )

  (:action cut_food
    :parameters (?f - food ?k - utensil ?l - location)
    :precondition (and (at ?f ?l) (at ?k ?l) (not (cut ?f)))
    :effect (cut ?f)
  )

  (:action cook_food
    :parameters (?f - food ?p - container ?s - location)
    :precondition (and (at ?f ?s) (at ?p ?s) (empty ?p))
    :effect (and (cooked ?f) (contains ?p ?f))
  )

  (:action clean_object
    :parameters (?o - object ?cs - cleaning_supplies ?l - location)
    :precondition (and (at ?o ?l) (at ?cs ?l) (dirty ?o))
    :effect (and (clean ?o) (not (dirty ?o)))
  )

  (:action dirty_object
    :parameters (?o - object)
    :precondition (clean ?o)
    :effect (dirty ?o)
  )
)
