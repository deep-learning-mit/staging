(define (domain bagel-preparation)
  (:requirements :strips)

  (:predicates
    (on ?x ?y)  ; Indicates ?x is on ?y
    (cut ?x)  ; Indicates ?x is cut
    (cooked ?x)  ; Indicates ?x is cooked
    (dirty ?x)  ; Indicates ?x is dirty
    (clean ?x)  ; Indicates ?x is clean
    (stove_on)  ; Indicates the stove is on
  )

  ; Objects: bagel, plate, knife, pan, stove, spatula, soap bottle, scrubber, sink

  (:action place_on_plate
    :parameters (?x)
    :precondition (and (not (on ?x plate)))
    :effect (on ?x plate)
  )

  (:action cut_bagel
    :parameters (?b ?k)
    :precondition (and (on ?b plate) (not (cut ?b)))
    :effect (and (cut ?b) (dirty ?k))
  )

  (:action cook_bagel
    :parameters (?b ?p)
    :precondition (and (cut ?b) (not (cooked ?b)) (stove_on))
    :effect (and (cooked ?b) (dirty ?p))
  )

  (:action clean_tool
    :parameters (?t)
    :precondition (and (dirty ?t))
    :effect (and (clean ?t) (not (dirty ?t)))
  )

  (:action turn_on_stove
    :precondition (not (stove_on))
    :effect (stove_on)
  )
)