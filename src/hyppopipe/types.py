from typing import Any, Self

NO_VALUE = object()


class HyppopipeError(Exception): ...


class Percentage(float):
    def __new__(cls, value: Any) -> Self:
        val = float(value)
        if not 0.0 <= val <= 1.0:
            raise ValueError("Invalid percentage. Value must be between 0 and 1.")
        return super().__new__(cls, val)


class PositiveInt(int):
    def __new__(cls, value: Any) -> Self:
        val = int(value)
        if val <= 0:
            raise ValueError("Invalid positive integer. Value must be positive.")
        return super().__new__(cls, val)


class NonNegativeInt(int):
    def __new__(cls, value: Any) -> Self:
        val = int(value)
        if val < 0:
            raise ValueError(
                "Invalid non-negative integer. Value must be non-negative."
            )
        return super().__new__(cls, val)
