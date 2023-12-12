(define (domain bagel-cutting)
  (:requirements :strips)
  (:predicates
    (plate-empty)
    (bowl-empty)
    (bagel-on-plate)
    (bagel-on-bowl)
    (knife-on-plate)
    (bagel-cut)
  )

  (:action place-bagel-on-plate
    :parameters ()
    :precondition (and (plate-empty) (bowl-empty))
    :effect (and (not (plate-empty)) (bagel-on-plate))
  )

  (:action place-knife-on-plate
    :parameters ()
    :precondition (plate-empty)
    :effect (and (not (plate-empty)) (knife-on-plate))
  )

  (:action cut-bagel
    :parameters ()
    :precondition (and (bagel-on-plate) (knife-on-plate))
    :effect (and (bagel-cut) (not (bagel-on-plate)) (not (knife-on-plate)) (plate-empty))
  )
)
