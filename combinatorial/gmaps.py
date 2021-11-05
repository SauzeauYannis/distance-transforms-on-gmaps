# AUTOGENERATED! DO NOT EDIT! File to edit: notebooks/00_gmaps.ipynb (unless otherwise specified).

__all__ = ['Marks', 'DualArray', 'nGmap']

# Cell

import numpy as np
import itertools
import logging
import logging_configuration


# get logger
logger = logging.getLogger("gmap_logger")
logging_configuration.set_logging()


class Marks:
    @property
    def m (self):
        return self._marks.shape[0]

    @property
    def free_marks(self):
        return self._free_marks

    def __init__ (self, m, d):
        self._marks      = np.zeros ((m,d), dtype=np.bool8)
        self._free_marks = {i for i in range (m)}

    def reserve_mark (self):
        m = self._free_marks.pop()
        self._free_marks -= {m}
        return m

    def free_mark (self,m):
        self._free_marks |= {m}

    def marked (self,m,d):
        return self._marks [m,d]

    def mark (self,m,d):
        self._marks [m,d] = True

    def unmark (self,m,d):
        self._marks [m,d] = False

    def mark_all (self,m):
        self._marks [m,:] = True

    def unmark_all (self,m):
        self._marks [m,:] = False

# Cell


class DualArray(np.ndarray):
    @property
    def D(self):
        return self[::-1]

    def __new__(cls, input_array):
        obj = np.asarray(input_array).view(cls)
        return obj

    def __array_finalize__(self, obj):
        # see InfoArray.__array_finalize__ for comments
        if obj is None: return
        self._marks      = getattr(obj, '_marks',      None)
        self._free_marks = getattr(obj, '_free_marks', None)


class nGmap(DualArray, Marks):
    """g-map based on indices"""

    @classmethod
    def _init_structures(cls, number_of_darts: int):
        logger.debug(f"The number of darts is: {number_of_darts}")
        # uint32 is sufficient for 1000x1000 images.
        dtype = np.int32 # I use a signed dtype for using negative values as markers (example, None type)
        if number_of_darts > (np.iinfo(dtype).max + 1):
            raise Exception(f"{dtype} is not sufficient to represent {number_of_darts} darts")

        # allocate distances array
        cls.distances = np.zeros(number_of_darts, dtype=dtype)
        logger.debug(f"distances array successfully initialized with shape {cls.distances.shape}"
                     f" and dtype {cls.distances.dtype}")

        # allocate labels array
        # int16 is sufficient to represent a label
        cls.image_labels = np.zeros(number_of_darts, dtype=np.int16)
        logger.debug(f"labels array successfully initialized with shape {cls.image_labels.shape}"
                     f" and dtype {cls.image_labels.dtype}")

        # allocate face identifiers array
        cls.face_identifiers = np.zeros(number_of_darts, dtype=dtype)
        logger.debug(f"face_identifiers array successfully initialized with shape {cls.face_identifiers.shape}"
                     f" and dtype {cls.face_identifiers.dtype}")

    def __init__ (self, array):
        super().__init__(8, self.shape[1])  # Create 8 marks for each (possible) dart

    @classmethod
    def n_by_d (cls, n, n_darts):
        """Initializes n-dimensional Gmap with n_darts isolated darts."""
        alphas = np.zeros ((n+1, n_darts), dtype=np.int)
        for i in range (n+1):
            alphas [i] = np.arange (n_darts)
        return cls.from_alpha_array (alphas)

    @classmethod
    def from_alpha_array (cls, a):
        """Constructs nGmap from an involution array.

        If the array of involutions `a` is not valid (in terms of G-maps definition) an excpetion is thrown.
        """
        if a.min() == 1:  # preparation for 1-based nGmaps (the smallest dart is 1 rather than 0)
            a = np.insert(a, 0, -1, axis=1)  # insert negative ones (invalid darts) as the 0-th column
        result = cls(a)
        if not result.is_valid:
            logging.critical('Have you passed an invalid involution array?')
            raise ValueError

        cls._init_structures(a.shape[1])

        return result

    @classmethod
    def from_string (cls, s):
        """Constructs a nGmap from an (n+1)-line string.

        See `combinatorial.zoo` for examples
        """
        n_lines = len (s.splitlines())
        arr = np.fromstring (s, sep = ' ', dtype = np.int).reshape (n_lines, -1)
        return cls.from_alpha_array (arr)

    @property
    def n(self):
        """Returns nGmap's dimension"""
        return self.shape[0] - 1

    @property
    def n_darts(self):
        """Returns the number of darts"""
        ### return (self[0] >= 0).sum()  # this would first create another big array of booleans
        return sum (1 for d in self.darts)

    def set_dart_distance(self, identifier: np.uint32, distance: np.uint32) -> None:
        self.distances[identifier] = distance

    @property
    def darts(self):
        """Generator to iterate thru all valid (non-negative alphas) darts"""
        for index in range (self.shape[1]):
            if self.a0(index) >= 0:
                yield index

    def all_dimensions_but_i (self, i=None):
        """Return a sorted sequence [0,...,n], without i, if 0 <= i <= n"""
        assert i is None or 0 <= i <= self.n
        return [j for j in range (self.n+1) if j != i]

    @property
    def all_dimensions(self):
        return self.all_dimensions_but_i()

    def set_ai (self, i, dart, new_dart):
        """Sets dart.alpha_i = new_dart"""
        assert 0 <= i <= self.n
        self [i,dart] = new_dart

    def ai (self, i, indices): return self[i,indices]  # TODO direct access
    def a0 (self,    indices): return self.ai(0,indices)
    def a1 (self,    indices): return self.ai(1,indices)
    def a2 (self,    indices): return self.ai(2,indices)
    def a3 (self,    indices): return self.ai(3,indices)
    def a4 (self,    indices): return self.ai(4,indices)

    @property
    def is_valid(self):
        """Checks validity, i.e., if a_i and (a_i  a_j) are involutions"""

        # check ai
        for i in self.all_dimensions:
            logging.debug (f'Involution check for  α{i}     ') #, end=' ')
            for dart in self.darts:
                if self.ai (i, self.ai (i, dart)) != dart:
                    logging.debug (f'broken at dart {dart}: α{i}.α{i} ({dart}) = {self.ai (i, self.ai (i, dart))} :(')
                    return False
            logging.debug ('passed.')

        # check ai.aj
        ij_pairs = itertools.combinations (self.all_dimensions, 2)
        for i,j in ij_pairs:
            logging.debug (f'Involution check for (α{i}.α{j}) ') #, end=' ')
            if j - i in {-1,+1}:
                logging.debug ('not required.')
                continue
            for dart in self.darts:
                aij = self.ai (i, self.ai (j, dart))
                if self.ai (i, self.ai (j, aij)) != dart:
                    logging.critical (f'broken at dart {dart}: (α{i} α{j})({dart}) = {self.ai (i, self.ai (j, dart))} :(')
                    return False
            logging.debug ('passed.')
        return True

    def orbit (self, sequence, d):
        """
        Orbit iterator
        For given dart and valid sequence of indices iterate over the orbit

        sequence example [0,2,3] to iterate around edge
        """

        m = self.reserve_mark()
        P = [d] # stack of darts to process
        self.mark (m,d) # mark the dart

        try:
            while len (P): # we still have some dart(s) to process
                cur = P.pop ()
                yield cur
                for j in sequence:
                    dd = self.ai (j, cur)
                    if not self.marked (m,dd):
                        self.mark (m,dd)
                        P.append (dd)
        finally:
            self.unmark_all (m)
            self.free_mark(m)

    def darts_of_i_cells(self, i=None):
        """Yields arbitrary dart for every i-cell (0<=i<=n) or for every connected component (i is None)"""

        m = self.reserve_mark()
        try:
            for d in self.darts:
                if not self.marked (m,d):
                    yield d
                    for dd in self.orbit(self.all_dimensions_but_i(i), d):  # i.e, in self.cell_i (i, d):
                        self.mark (m,dd)
        finally:
            self.unmark_all (m)
            self.free_mark(m)

    def all_i_cells (self, i=None):
        """For each i-cell (connected component) yield a list of its darts"""
        logging.debug (f'Listing {i}-cells:')
        for counter, d in enumerate (self.darts_of_i_cells(i)):
            cell = list (self.cell_i (i, d))  # compute orbit from d
            logging.debug (f'\t#{counter+1:2d}: {cell}')
            yield cell

    def all_connected_components (self):
        """For each connected component yields a set of its darts"""
        return self.all_i_cells()

    def cell_i (self, i, dart):
        """iterator over i-cell of a given dart"""
        return self.orbit (self.all_dimensions_but_i(i), dart)

    def cell_0 (self, dart): return self.cell_i (0, dart)
    def cell_1 (self, dart): return self.cell_i (1, dart)
    def cell_2 (self, dart): return self.cell_i (2, dart)
    def cell_3 (self, dart): return self.cell_i (3, dart)
    def cell_4 (self, dart): return self.cell_i (4, dart)

    # number of i-cells and connected components

    def no_i_cells (self, i=None):
        """
        Counts
            i-cells,             if 0 <= i <= n
            connected components if i is None
        """
        assert i is None or 0 <= i <= self.n
        # return more_itertools.ilen (self.darts_of_i_cells(i))
        return sum ((1 for d in self.darts_of_i_cells(i)))

    @property
    def no_0_cells (self): return self.no_i_cells (0)
    @property
    def no_1_cells (self): return self.no_i_cells (1)
    @property
    def no_2_cells (self): return self.no_i_cells (2)
    @property
    def no_3_cells (self): return self.no_i_cells (3)
    @property
    def no_4_cells (self): return self.no_i_cells (4)
    @property
    def no_ccs     (self): return self.no_i_cells ( )

    def _link (self, i, dart1, dart2):
        """i-link dart1 and dart2. Use with care"""
        self.set_ai(i,dart1, dart2)
        self.set_ai(i,dart2, dart1)

    def print_alpha_table(self, tableseparators = '=-='):
        """
        Print
        """
        sepT, sepM, sepB = tableseparators
        print ((7 + 3*self.n_darts)*sepT)
        print (  ' D# | ', end=' ')
        for d in self.darts:
            print (f'{d:2} ', end='')
        print ()
        print ((7 + 3*self.n_darts)*sepM)
        for i in self.all_dimensions:
            print (f' α{i} | ', end=' ')
            for d in self.darts:
                print (f'{self.ai(i,d):2} ', end='')
            print ()
        print ((7 + 3*self.n_darts)*sepB)


    def summary_string(self):
        """Returns a summary string"""

        if self.n < 0: return "Void gmap"

        s = f'{self.n}-gMap of {self.n_darts} darts:\n'
        for i in self.all_dimensions_but_i():
            s += f'  # {i}-cells: {self.no_i_cells(i)}\n'
        s += f'  # ccs    : {self.no_ccs}\n'
        return s

    def __str__(self):
        return self.summary_string()

    def __repr__(self):
        pass
        return self.summary_string()


    # -----------------------
    # incidence and adjacency

    def incident (self, i, d1, j, d2):
        """True if i-cell of d is incident with j-cell of d2)"""
        # simply a check if the intersection of the two respective orbits is nonempty
        # uses sets,  `&` for interesection , len() for cardinality)
        # TODO using set for large orbits may be memory inefficient

        i_cell_of_d1 = set (self.cell_i (i,d1))
        j_cell_of_d2 = set (self.cell_i (j,d2))
        return len (i_cell_of_d1 & j_cell_of_d2) > 0

    def adjacent (self,i,d1,d2):
        """True if i-cell of d is adjacent to i-cell of d2.

        For adjacency we have the same $i$ for both darts.
        Algorithm:
            1. get both i-cells
            2. make $\alpha_i$ of all elements of the first orbit
            3. check if they have an intersection with the second orbit
        """

        i_cell_of_d1 = self.cell_i (i,d1)  # iterator over i-cell of d1
        i_cell_of_d2 = self.cell_i (i,d2)  # iterator over i-cell of d2
        s_cell_of_d1 = set(i_cell_of_d1)   # set of darts of i-cell of d1
        # TODO  query 'in' python set could be O(1) but creating set requires additional memory
        for d in i_cell_of_d2:
            ai_d = self.ai(i,d)
            if ai_d in s_cell_of_d1:  # TODO do we really need to test if d1 \neq d2?!
                logging.debug (f'a{i}({d}) == {ai_d} which belongs to {i}-cell of dart {d1}, so these {i}-cells are adjacent')
                return True
        return False

    # -----------------------
    # contraction and removal

    def _remove_dart (self,d):
        # self [:,d] = -1    # this would be a direct access
        for i in self.all_dimensions:
            self.set_ai(i,d,-1)

    def _is_i_removable_or_contractible(self, i, dart, rc):
        """
        Test if an i-cell of dart is removable/contractible:

        d  ... dart
        i  ... i-cell
        rc ... +1 => removable test, -1 => contractible test
        """
        # TODO: assert dart ist valid
        assert 0 <= i <= self.n
        assert rc in {-1, +1}

        if rc == +1:  # removable test
            if i == self.n  : return False
            if i == self.n-1: return True
        if rc == -1:  # contractible test
            if i == 0: return False
            if i == 1: return True

        for d in self.cell_i(i, dart):
            if self.ai (i+rc, self.ai(i+rc+rc, d)) != self.ai (i+rc+rc, self.ai (i+rc, d)):
                return False
        return True

    def is_i_removable(self, i, dart):
        """True if i-cell of dart can be removed"""
        return self._is_i_removable_or_contractible(i, dart, rc=+1)

    def is_i_contractible(self, i, dart):
        """True if i-cell of dart can be contracted"""
        return self._is_i_removable_or_contractible(i, dart, rc=-1)

    def _i_remove_contract(self, i, dart, rc, skip_check=False):
        """
        Remove / contract an i-cell of dart
        d  ... dart
        i  ... i-cell
        rc ... +1 => remove, -1 => contract
        skip_check ... set to True if you are sure you can remove / contract the i-cell
        """
        logging.debug (f'{"Remove" if rc == 1 else "Contract"} {i}-Cell of dart {dart}')

        if not skip_check:
            assert self._is_i_removable_or_contractible(i, dart, rc),\
                f'{i}-cell of dart {dart} is not {"removable" if rc == 1 else "contractible"}!'

        i_cell = set(self.cell_i(i, dart))  # mark all the darts in ci(d)
        logging.debug (f'\n{i}-cell to be removed {i_cell}')
        for d in i_cell:
            d1 = self.ai (i,d) # d1 ← d.Alphas[i];
            if d1 not in i_cell:  # if not isMarkedNself(d1,ma) then
                # d2 ← d.Alphas[i + 1].Alphas[i];
                d2 = self.ai (i+rc,d)
                d2 = self.ai (i   ,d2)
                while d2 in i_cell: # while isMarkedNself(d2,ma) do
                    # d2 ← d.Alphas[i + 1].Alphas[i];
                    d2 = self.ai (i+rc,d2)
                    d2 = self.ai (i   ,d2)
                logging.debug (f'Modifying alpha_{i} of dart {d1} from {self.ai (i,d1)} to {d2}')

                self.set_ai(i,d1,d2) # d1.Alphas[i] ← d2;
                # Update the face identifier value of the removed dart
                # I choose d1 but I can also choose d2.
                # I have to memorize to which face is now associated d when it is removed
                # I identify the face by one of his darts
                self.face_identifiers[d] = d1
            else:
                # I need to find a value to wich associate d.
                # d1 is not suitable because is in the i-cell
                # I can find a dart in the i+1 cell to associate the value that is not in the i-cell to remove
                found = False
                for d1 in self.cell_i(i + 1, d):
                    if d1 not in i_cell:
                        self.face_identifiers[d] = d1
                        found = True
                        break
                if not found:
                    # try with the other cell
                    d_other_cell = self.ai(i+1, d) # d1 ← d.Alphas[i];
                    for d1 in self.cell_i(i + 1, d_other_cell):
                        if d1 not in i_cell:
                            self.face_identifiers[d] = d1
                            found = True
                            break
                if not found:
                    logger.debug(f"Not found a suitable dart for dart: {d}")

        for d in i_cell:  # foreach dart d' ∈ ci(d) do
            self._remove_dart(d)  # remove d' from gm.Darts;

    def _remove(self, i, dart, skip_check=False):
        """Remove i-cell of dart"""
        self._i_remove_contract(i, dart, rc=+1, skip_check=skip_check)

    def _contract(self, i, dart, skip_check=False):
        """Contract i-cell of dart"""
        self._i_remove_contract(i, dart, rc=-1, skip_check=skip_check)

    def remove_0_cell(self, dart): self._remove(0, dart)
    def remove_1_cell(self, dart): self._remove(1, dart)
    def remove_2_cell(self, dart): self._remove(2, dart)
    def contract_1_cell(self, dart): self._contract(1, dart)
    def contract_2_cell(self, dart): self._contract(2, dart)
    def contract_3_cell(self, dart): self._contract(3, dart)

    def remove_vertex  (self, dart): self._remove(0, dart)
    def remove_edge    (self, dart): self._remove(1, dart)
    def remove_face    (self, dart): self._remove(2, dart)
    def contract_edge  (self, dart): self._contract(1, dart)
    def contract_face  (self, dart): self._contract(2, dart)
    def contract_volume(self, dart): self._contract(3, dart)

    def sew_seq(self, i):
        """
        Orbit indices to be used in the sewing operations.
        This sequence will be empty for the following cases
            - n=0: i=0
            - n=1: i=0 or i=1
            - n=2:        i=1
        (0, ..., i - 2, i + 2, ..., n)
        """
        return itertools.chain(range(0, i - 1), range(i + 2, self.n + 1))

    def sew_no_assert(self, d1, d2, i):
        """
        i-sew darts d1, d2 (and the necessary orbits), w/o checking if the operation in sewable
        Args:
            d1: first  dart to sew
            d2: second dart to sew
            i:  0...n

        """
        indices = list (self.sew_seq(i))
        for e1, e2 in zip(self.orbit(indices, d1), self.orbit(indices, d2)):
            self._link(i, e1, e2)