(define (domain cut-bagel)
  (:requirements :strips :typing :equality)
  (:types
    plate
    bowl
    knife
    bagel
  )

  (:predicates
    (isCut ?b - bagel)
    (isWhole ?b - bagel)
    (onPlate ?b - bagel)
    (onBowl ?b - bagel)
    (clean ?k - knife)
    (dirty ?k - knife)
    (hasKnife)
  )

  (:action pick-up-knife
    :precondition (clean ?k - knife)
    :effect (and
      (not (clean ?k - knife))
      (hasKnife)
      (dirty ?k - knife)
    )
  )

  (:action put-down-knife
    :precondition (and (dirty ?k - knife) (hasKnife))
    :effect (and
      (not (dirty ?k - knife))
      (not (hasKnife))
      (clean ?k - knife)
    )
  )

  (:action cut-bagel
    :precondition (and (onPlate ?b - bagel) (isWhole ?b - bagel) (hasKnife))
    :effect (and
      (not (isWhole ?b - bagel))
      (isCut ?b - bagel)
    )
  )

  (:action move-bagel-to-plate
    :precondition (onBowl ?b - bagel)
    :effect (and
      (not (onBowl ?b - bagel))
      (onPlate ?b - bagel)
    )
  )

  (:action move-bagel-to-bowl
    :precondition (onPlate ?b - bagel)
    :effect (and
      (not (onPlate ?b - bagel))
      (onBowl ?b - bagel)
    )
  )
)
