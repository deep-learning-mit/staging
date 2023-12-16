(define (domain kitchen-cleanup)
  (:requirements :strips :typing)

  ;; Define the types of objects we have in our domain
  (:types
    utensil plate food
  )

  ;; Define the predicates that describe the states in the domain
  (:predicates
    (is_dirty ?p - plate)
    (has_crumbs ?p - plate)
    (is_clean ?p - plate)
    (is_cut ?f - food)
    (is_uncut ?f - food)
    (is_on_table ?o)
    (is_in_sink ?o)
  )

  ;; Actions
  (:action clean_plate
    :parameters (?p - plate)
    :precondition (and (is_dirty ?p) (is_on_table ?p))
    :effect (and (is_clean ?p) (not (is_dirty ?p)) (not (has_crumbs ?p)))
  )

  (:action pick_up_crumbs
    :parameters (?p - plate)
    :precondition (and (has_crumbs ?p) (is_on_table ?p))
    :effect (not (has_crumbs ?p))
  )

  (:action move_to_sink
    :parameters (?o - object)
    :precondition (is_on_table ?o)
    :effect (and (is_in_sink ?o) (not (is_on_table ?o)))
  )

  (:action cut_food
    :parameters (?f - food)
    :precondition (and (is_uncut ?f) (is_on_table ?f))
    :effect (and (is_cut ?f) (not (is_uncut ?f)))
  )

  ;; We can add more actions for other tasks like slicing the bagel, 
  ;; putting away the utensil, etc.
)
