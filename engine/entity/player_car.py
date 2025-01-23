from engine.entity.car import Car


class PlayerCar(Car):
    """
    A class representing the player car.
    """

    def _get_input(self) -> Car.Input:
        return Car.Input(1.0, 1.0)
