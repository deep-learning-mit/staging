(define (domain kitchen)
  (:requirements :typing)
  (:types item surface food)

  ; Predicates describing the state of the world
  (:predicates
    (on ?x - item ?y - surface)
    (empty ?x - surface)
    (cut ?x - food)
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
    :precondition (and (on ?bagel ?surface) (on ?knife ?surface) (not (cut ?bagel)))
    :effect (and
              (cut ?bagel)
            )
  )
)