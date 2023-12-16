(define (domain kitchen-task)
  (:requirements :strips :typing)
  (:types
    food - object
    container - object
    utensil - object
    appliance - object
  )
  (:predicates
    (isCut food)
    (isCooked food)
    (isSpread food)
    (isEmpty container)
    (contains container food)
    (isHot appliance)
    (isClean utensil)
    (isDirty utensil)
    (on appliance container)
  )
  (:action cut
    :parameters (?f - food ?k - utensil)
    :precondition (and (not (isCut ?f)) (isClean ?k))
    :effect (and (isCut ?f) (isDirty ?k))
  )
  (:action cook
    :parameters (?f - food ?p - appliance ?c - container)
    :precondition (and (not (isCooked ?f)) (isHot ?p) (contains ?c ?f))
    :effect (isCooked ?f)
  )
  (:action spread
    :parameters (?f - food ?u - utensil)
    :precondition (and (isCut ?f) (isClean ?u))
    :effect (isSpread ?f)
  )
  (:action heat
    :parameters (?a - appliance)
    :precondition (not (isHot ?a))
    :effect (isHot ?a)
  )
  (:action clean
    :parameters (?u - utensil)
    :precondition (isDirty ?u)
    :effect (isClean ?u)
  )
  (:action put
    :parameters (?f - food ?c - container)
    :precondition (not (contains ?c ?f))
    :effect (and (contains ?c ?f) (isEmpty ?c))
  )
  (:action remove
    :parameters (?f - food ?c - container)
    :precondition (contains ?c ?f)
    :effect (not (contains ?c ?f))
  )
)