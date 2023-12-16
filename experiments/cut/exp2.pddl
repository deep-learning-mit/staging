(define (domain kitchen-setup)
  (:requirements :strips :typing)
  (:types 
    food - object
    container - object
    utensil - object
  )

  (:predicates
    (isPlateEmpty ?plate - container)
    (isFoodOnPlate ?food - food ?plate - container)
    (isUtensilClean ?utensil - utensil)
    (isFoodInBowl ?food - food ?bowl - container)
  )

  (:action put_food_on_plate
    :parameters (?food - food ?plate - container)
    :precondition (and (not (isFoodOnPlate ?food ?plate)) (isPlateEmpty ?plate))
    :effect (and (isFoodOnPlate ?food ?plate) (not (isPlateEmpty ?plate)))
  )

  (:action remove_food_from_plate
    :parameters (?food - food ?plate - container)
    :precondition (isFoodOnPlate ?food ?plate)
    :effect (and (not (isFoodOnPlate ?food ?plate)) (isPlateEmpty ?plate))
  )

  (:action clean_utensil
    :parameters (?utensil - utensil)
    :precondition (not (isUtensilClean ?utensil))
    :effect (isUtensilClean ?utensil)
  )

  (:action take_food_from_bowl
    :parameters (?food - food ?bowl - container)
    :precondition (isFoodInBowl ?food ?bowl)
    :effect (not (isFoodInBowl ?food ?bowl))
  )
)
