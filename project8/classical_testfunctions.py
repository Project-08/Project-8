import numpy as np


class testcases:

    def get_error(
            self,
            u_num: np.ndarray[tuple[int], np.dtype[np.float64]],
            u_exact: np.ndarray[tuple[int], np.dtype[np.float64]]
            ) -> np.float64:
        squares = (u_num-u_exact)**2
        mean = np.mean(squares)
        root = np.sqrt(mean)
        return np.float64(root)


class cases_3d(testcases):

    def get_analytical(
            self,
            steps: int,
            stepsize: float
            ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

        result = np.zeros((steps-1)**3)
        for z in range(1, steps):
            for y in range(1, steps):
                for x in range(1, steps):
                    result[self._get_index(x, y, z, steps)] =\
                        self._analytical_solution(x, y, z, stepsize)
        return result

    def _get_index(self, x: int, y: int, z: int, steps: int) -> int:
        return (x-1)+(steps-1)*(y-1)+(steps-1)*(steps-1)*(z-1)

    def get_9_point_source_func(
            self,
            steps: int,
            stepsize: float
            ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = np.zeros((steps-1)**3)
        for z in range(1, steps):
            for y in range(1, steps):
                for x in range(1, steps):
                    result[self._get_index(x, y, z, steps)] =\
                        self._source_func(x, y, z, stepsize)
                    # Boundary Condition:
                    if (x == 1):
                        result[self._get_index(x, y, z, steps)]\
                            += self.x0(x, y, z, stepsize)/(stepsize**2)
                    if (x == (steps-1)):
                        result[self._get_index(x, y, z, steps)]\
                            += self.x1(x, y, z, stepsize)/(stepsize**2)
                    if (y == 1):
                        result[self._get_index(x, y, z, steps)]\
                            += self.y0(x, y, z, stepsize)/(stepsize**2)
                    if (y == (steps-1)):
                        result[self._get_index(x, y, z, steps)]\
                            += self.y1(x, y, z, stepsize)/(stepsize**2)
                    if (z == 1):
                        result[self._get_index(x, y, z, steps)]\
                            += self.z0(x, y, z, stepsize)/(stepsize**2)
                    if (z == (steps-1)):
                        result[self._get_index(x, y, z, steps)]\
                            += self.z1(x, y, z, stepsize)/(stepsize**2)

        return result

    def _source_func(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def _analytical_solution(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def x0(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def x1(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def y0(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def y1(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def z0(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def z1(
            self,
            x: int,
            y: int,
            z: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)


class cases_2d(testcases):

    def get_analytical(
            self,
            steps: int,
            stepsize: float
            ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:

        result = np.zeros((steps-1)**2)
        for y in range(1, steps):
            for x in range(1, steps):
                result[self._get_index(x, y, steps)] =\
                    self._analytical_solution(x, y, stepsize)
        return result

    def _get_index(self, x: int, y: int, steps: int) -> int:
        return (x-1)+(steps-1)*(y-1)

    def get_9_point_source_func(
            self,
            steps: int,
            stepsize: float
            ) -> np.ndarray[tuple[int], np.dtype[np.float64]]:
        result = np.zeros((steps-1)**2)
        for y in range(1, steps):
            for x in range(1, steps):
                result[self._get_index(x, y, steps)] =\
                    self._source_func(x, y, stepsize)
                # Boundary Condition:
                if (x == 1):
                    result[self._get_index(x, y, steps)]\
                        += self.x0(x, y, stepsize)/(stepsize**2)
                if (x == (steps-1)):
                    result[self._get_index(x, y, steps)]\
                        += self.x1(x, y, stepsize)/(stepsize**2)
                if (y == 1):
                    result[self._get_index(x, y, steps)]\
                        += self.y0(x, y, stepsize)/(stepsize**2)
                if (y == (steps-1)):
                    result[self._get_index(x, y, steps)]\
                        += self.y1(x, y, stepsize)/(stepsize**2)
        return result

    def _source_func(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def _analytical_solution(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def x0(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def x1(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def y0(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)

    def y1(
            self,
            x: int,
            y: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(0)


# example from https://doi.org/10.1016/j.apm.2011.11.078
class example_1_2d(cases_2d):

    def _source_func(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64(np.sin(np.pi*x)*np.sin(np.pi*y))

    def _analytical_solution(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64(-np.sin(np.pi*x)*np.sin(np.pi*y)/(2*np.pi*np.pi))


# example from https://doi.org/10.1016/j.procs.2010.04.041
class example_2_2d(cases_2d):
    def _source_func(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64((x*x + y*y)*np.exp(x*y))

    def _analytical_solution(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64(np.exp(x*y))

    def x0(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(1)

    def x1(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        y = q*stepsize
        return np.float64(np.exp(y))

    def y0(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(1)

    def y1(
            self,
            p: int,
            q: int,
            stepsize: float
            ) -> np.float64:
        x = p*stepsize
        return np.float64(np.exp(x))


# form 10.4236/ajcm.2011.14035:
# Examples 3, 4, 5 and 6
class example_3_3d(cases_3d):
    def _source_func(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(2*(x*y + x*z + y*z))

    def _analytical_solution(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(x*y*z*(x+y+z))

    def x1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        y, z = q*stepsize, r*stepsize
        return np.float64(y*z*(1+y+z))

    def y1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, z = p*stepsize, r*stepsize
        return np.float64(x*z*(x+1+z))

    def z1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, = p*stepsize, q*stepsize
        return np.float64(x*y*(x+y+1))


class example_4_3d(cases_3d):
    def _source_func(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        return np.float64(6)

    def _analytical_solution(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(x*x + y*y + z*z)

    def x0(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        y, z = q*stepsize, r*stepsize
        return np.float64(y*y + z*z)

    def x1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        y, z = q*stepsize, r*stepsize
        return np.float64(1 + y*y + z*z)

    def y0(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, z = p*stepsize, r*stepsize
        return np.float64(x*x + z*z)

    def y1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, z = p*stepsize, r*stepsize
        return np.float64(x*x + 1 + z*z)

    def z0(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64(x*x + y*y)

    def z1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y = p*stepsize, q*stepsize
        return np.float64(x*x + y*y + 1)


class example_5_3d(cases_3d):
    def _analytical_solution(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(x*y*np.sin(np.pi*z))

    def _source_func(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(-np.pi*np.pi*x*y*np.sin(np.pi*z))

    def x1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        y, z = q*stepsize, r*stepsize
        return np.float64(y*np.sin(np.pi*z))

    def y1(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, z = p*stepsize, r*stepsize
        return np.float64(x*np.sin(np.pi*z))


class example_6_3d(cases_3d):
    def _source_func(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(
            -3*np.pi*np.pi
            * np.sin(np.pi*x)
            * np.sin(np.pi*y)
            * np.sin(np.pi*z)
            )

    def _analytical_solution(
            self,
            p: int,
            q: int,
            r: int,
            stepsize: float
            ) -> np.float64:
        x, y, z = p*stepsize, q*stepsize, r*stepsize
        return np.float64(
            np.sin(np.pi*x)
            * np.sin(np.pi*y)
            * np.sin(np.pi*z)
            )
