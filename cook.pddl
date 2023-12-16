(define (domain kitchen)
  (:requirements :typing)
  (:types item surface food)

  ; Predicates describing the state of the world
  (:predicates
    (hot ?x - surface)
    (on ?x - item ?y - surface)
    (empty ?x - surface)
    (isPan ?x - surface)
    (isStove ?x - surface)
    (cut ?x - food)
    (cooked ?x - food)
  )

  ; Action to move an item from one surface to another
  (:action move
    :parameters (?item - item ?from - surface ?to - surface)
    :precondition (and (on ?item ?from))
    :effect (and
              (not (on ?item ?from))
              (on ?item ?to)
    )
  )

  ; Action to cut the bagel
  (:action cut_bagel
    :parameters (?bagel - food ?knife - item ?surface - surface)
    :precondition (and (on ?bagel ?surface) (on ?knife ?surface) (not (isStove ?surface)) (not (cut ?bagel)))
    :effect (and
              (cut ?bagel)
            )
  )

  (:action turn_on_stove
    :parameters (?stove - surface)
    :precondition (and (not (hot ?stove)))
    :effect (and 
                (hot ?stove)
            )
  )

  (:action turn_off_stove
    :parameters (?stove - surface)
    :precondition (and (hot ?stove))
    :effect (and 
                (not (hot ?stove))
            )
  )

  ; Action to cook the bagel
  (:action cook_bagel
    :parameters (?bagel - food ?stove - surface ?pan - surface)
    :precondition (and (
        (cut ?bagel)
        (hot ?stove)
        (isStove ?stove)
        (isPan ?pan)
        (on ?bagel ?pan)
        (on ?pan ?stove)
        (not (cooked ?bagel))
    ))
    :effect (and (
      (cooked ?bagel)
    ))
  )
)