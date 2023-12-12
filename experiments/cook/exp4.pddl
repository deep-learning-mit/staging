(define (domain cooking-bagel)
  (:requirements :strips)
  (:predicates
    (isBagelCut)
    (isStoveOn)
    (isBagelOnPlate)
    (isBagelOnStove)
    (isBagelInPan)
    (isKnifeOnPlate)
    (isSpatulaOnPlate)
  )

  (:action cut-bagel
    :parameters (?bagel ?knife ?plate)
    :precondition (and (isBagelOnPlate ?bagel) (isKnifeOnPlate ?knife))
    :effect (and (isBagelCut ?bagel) (not (isBagelOnPlate ?bagel)))
  )

  (:action turn-on-stove
    :parameters (?stove)
    :precondition (not (isStoveOn ?stove))
    :effect (isStoveOn ?stove)
  )

  (:action place-bagel-in-pan
    :parameters (?bagel ?pan)
    :precondition (and (isBagelCut ?bagel) (not (isBagelInPan ?bagel)))
    :effect (isBagelInPan ?bagel)
  )

  (:action cook-bagel
    :parameters (?bagel ?pan ?stove)
    :precondition (and (isStoveOn ?stove) (isBagelInPan ?bagel))
    :effect (isBagelOnStove ?bagel)
  )
)