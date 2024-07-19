#ifndef SURFACE_CODE_QUBIT_HPP
#define SURFACE_CODE_QUBIT_HPP

#include <map>
#include <vector>
#include <cstdint>

using std::size_t;

enum surface_role {data, a_mx, a_mz, empty};

struct surface_coord{
  int x;
  int y;

  surface_coord(int x_in, int y_in);
};

struct role_index_pair{
  surface_role role;
  size_t index;

  role_index_pair();
  role_index_pair(const role_index_pair& pair_in);
  role_index_pair(surface_role role_in, size_t index_in);
};


// Technically this is a "rotated" surface code
class SurfaceCodeQubit{
  // everything public for now
  public:
    // primary goal is to generate surface code and parity check matrix
    // keep track of data qubits, mz ancillas, mx ancillas
    // and the 2d grid.
    // Want a mapping from qubit index -> 2d coord as well as
    // inverse map 2d coord -> qubit index
    uint32_t distance;
    std::vector<surface_coord> data_coords;
    std::vector<surface_coord> x_stab_coords;
    std::vector<surface_coord> z_stab_coords;
    std::map<surface_coord, size_t> data_indices;
    std::map<surface_coord, size_t> x_stab_indices;
    std::map<surface_coord, size_t> z_stab_indices;
    std::map<surface_coord, role_index_pair> grid_map;
    // In surface code, can have weight 2 or weight 4 stabs
    // So {x,z}_stabilizer[i].size() == 2 || 4
    std::vector<std::vector<size_t>> x_stabilizers;
    std::vector<std::vector<size_t>> z_stabilizers;

    SurfaceCodeQubit(uint32_t distance_in);

    void generate_grid();
    void generate_stabilizers();
    void print_qubit_coords() const;
    void print_qubit_indices() const;
    void print_grid_map() const;
    void print_stabilizers() const;
};

#endif // SURFACE_CODE_QUBIT_HPP


