#include "SurfaceCodeQubit.hpp"
#include <cstdio>
#include <algorithm>

surface_coord::surface_coord(int x_in, int y_in):
  x(x_in), y(y_in){
  }

surface_coord operator+(const surface_coord& lhs, const surface_coord& rhs){
  return {lhs.x + rhs.x, lhs.y + rhs.y};
}

surface_coord operator-(const surface_coord& lhs, const surface_coord& rhs){
  return {lhs.x - rhs.x, lhs.y - rhs.y};
}

bool operator==(const surface_coord& lhs, const surface_coord& rhs){
  return lhs.x == rhs.x && lhs.y == rhs.y;
}

bool operator<(const surface_coord& lhs, const surface_coord& rhs){
  if(lhs.x != rhs.x){
    return lhs.x < rhs.x;
  }
  return lhs.y < rhs.y;
}

role_index_pair::role_index_pair(){
}

role_index_pair::role_index_pair(const role_index_pair& pair_in):
  role(pair_in.role), index(pair_in.index){
  }

role_index_pair::role_index_pair(surface_role role_in, size_t index_in):
  role(role_in), index(index_in){
  }


// Defining the order of the CNOTs
// at the circuit level
void SurfaceCodeQubit::generate_stabilizers(){
  std::vector<surface_coord> order{
    {1, 1}, {1, -1}, {-1, 1}, {-1, -1}};

  for(size_t xi = 0; xi < x_stab_coords.size(); ++xi){
    std::vector<size_t> current_stab;
    // max weight stabilizer is 4
    for(const surface_coord& delta : order){
      surface_coord trial_coord = x_stab_coords[xi] + delta;
      // if data qubit is at trial coordinate, add to stabilizer
      if( data_indices.find(trial_coord) != data_indices.end() ){
        current_stab.push_back(data_indices[trial_coord]);
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    x_stabilizers.push_back(current_stab);
  }

  for(size_t zi = 0; zi < z_stab_coords.size(); ++zi){
    std::vector<size_t> current_stab;
    // max weight stabilizer is 4
    for(const surface_coord& delta : order){
      surface_coord trial_coord = z_stab_coords[zi] + delta;
      // if data qubit is at trial coordinate, add to stabilizer
      if( data_indices.find(trial_coord) != data_indices.end() ){
        current_stab.push_back(data_indices[trial_coord]);
      }
    }
    std::sort(current_stab.begin(), current_stab.end());
    z_stabilizers.push_back(current_stab);
  }

  return;
}

void SurfaceCodeQubit::print_stabilizers() const{
  for(size_t s_i = 0; s_i < x_stabilizers.size(); ++s_i){
    printf("s[%zu]: ", s_i);
    for(size_t op_i = 0; op_i < x_stabilizers[s_i].size(); ++op_i){
      printf("X%zu ", x_stabilizers[s_i][op_i]);
    }
    printf("\n");
  }
  size_t offset = x_stabilizers.size();
  for(size_t s_i = 0; s_i < z_stabilizers.size(); ++s_i){
    printf("s[%zu]: ", s_i + offset);
    for(size_t op_i = 0; op_i < z_stabilizers[s_i].size(); ++op_i){
      printf("Z%zu ", z_stabilizers[s_i][op_i]);
    }
    printf("\n");
  }

  return;
}

void SurfaceCodeQubit::print_qubit_coords() const{
  printf("%zu data qubits:\n", data_coords.size());
  for(size_t i = 0; i < data_coords.size(); ++i){
    printf("d[%zu] @ (%d, %d)\n",i, data_coords[i].x, data_coords[i].y);
  }
  printf("%zu mx ancilla qubits:\n", x_stab_coords.size());
  for(size_t i = 0; i < x_stab_coords.size(); ++i){
    printf("amx[%zu] @ (%d, %d)\n",i, x_stab_coords[i].x, x_stab_coords[i].y);
  }
  printf("%zu mz ancilla qubits:\n", z_stab_coords.size());
  for(size_t i = 0; i < z_stab_coords.size(); ++i){
    printf("amz[%zu] @ (%d, %d)\n",i, z_stab_coords[i].x, z_stab_coords[i].y);
  }
  return;
}

void SurfaceCodeQubit::print_qubit_indices() const{
  printf("%zu data qubits:\n", data_indices.size());
  for(const auto &[k, v] : data_indices){
    printf("@(%d,%d): d[%zu]\n", k.x, k.y, v);
  }
  printf("%zu mx ancilla qubits:\n", x_stab_indices.size());
  for(const auto &[k, v] : x_stab_indices){
    printf("@(%d,%d): amx[%zu]\n", k.x, k.y, v);
  }
  printf("%zu mz ancilla qubits:\n", z_stab_indices.size());
  for(const auto &[k, v] : z_stab_indices){
    printf("@(%d,%d): amz[%zu]\n", k.x, k.y, v);
  }
}

void SurfaceCodeQubit::print_grid_map() const{
  int width = 1;
  printf("Visualization of the surface code:\n");
  uint32_t grid_max = distance*2;
  for(int y = 0; y <= grid_max; ++y){
    for(int x = 0; x <= grid_max; ++x){
      surface_coord coord(x,y);
      if(grid_map.find(coord) == grid_map.end()){
        printf("%-*s",3+width, " ");
        continue;
      }
      role_index_pair q = grid_map.at(coord);
      switch(q.role){
        case data:
          printf("d%-2zu ", q.index);
          break;
        case a_mx:
          printf("mx%-2zu", q.index);
          break;
        case a_mz:
          printf("mz%-2zu", q.index);
          break;
        case empty:
          printf("e[%zu]", q.index);
        default:
          printf("invalid");
          break;
      }
    }
    printf("\n");
  }
  return;
}

void SurfaceCodeQubit::generate_grid(){
  uint32_t grid_max = distance*2;
  for(int y = 0; y <= grid_max; ++y){
    for(int x = 0; x <= grid_max; ++x){
      bool bot_or_top = (y == 0) || (y == grid_max);
      bool left_or_right = (x == 0) || (x == grid_max);
      bool x_parity = x % 2;
      bool y_parity = y % 2;
      bool x4_div = !(x % 4);
      bool y4_div = !(y % 4);
      if( x_parity && y_parity ){
        data_coords.push_back(surface_coord(x,y));
      } else if( x_parity || y_parity ){
      } else {
        if( bot_or_top && left_or_right ){
        } else if( (y4_div != x4_div) && !left_or_right ){
          x_stab_coords.push_back(surface_coord(x,y));
        } else if( (y4_div != x4_div) && !bot_or_top ){
        } else if( bot_or_top ){
        } else {
          z_stab_coords.push_back(surface_coord(x,y));
        }
      }
    }
  }

  std::sort(data_coords.begin(), data_coords.end());
  std::sort(x_stab_coords.begin(), x_stab_coords.end());
  std::sort(z_stab_coords.begin(), z_stab_coords.end());
  for(int i = 0; i < data_coords.size(); ++i){
    data_indices[data_coords[i]] = i;
    grid_map[data_coords[i]] = role_index_pair(surface_role::data, (size_t)i);
  }
  for(int i = 0; i < x_stab_coords.size(); ++i){
    x_stab_indices[x_stab_coords[i]] = i;
    grid_map[x_stab_coords[i]] = role_index_pair(surface_role::a_mx, (size_t)i);
  }
  for(int i = 0; i < z_stab_coords.size(); ++i){
    z_stab_indices[z_stab_coords[i]] = i;
    grid_map[z_stab_coords[i]] = role_index_pair(surface_role::a_mz, (size_t)i);
  }
  return;
}

void generate_and_print_grid(uint32_t distance){
  uint32_t grid_max = distance*2;
  for(int y = 0; y <= grid_max; ++y){
    for(int x = 0; x <= grid_max; ++x){
      bool bot_or_top = (y == 0) || (y == grid_max);
      bool left_or_right = (x == 0) || (x == grid_max);
      bool x_parity = x % 2;
      bool y_parity = y % 2;
      bool x4_div = !(x % 4);
      bool y4_div = !(y % 4);
      if( x_parity && y_parity ){
        printf("d");
      } else if( x_parity || y_parity ){
        printf(" ");
      } else {
        if( bot_or_top && left_or_right ){
          printf(" ");
        } else if( (y4_div != x4_div) && !left_or_right ){
          printf("x");
        } else if( (y4_div != x4_div) && !bot_or_top ){
          printf(" ");
        } else if( bot_or_top ){
          printf(" ");
        } else {
          printf("z");
        }
      }
    }
    printf("\n");
  }
  return;
}

SurfaceCodeQubit::SurfaceCodeQubit(uint32_t distance_in):
  distance(distance_in){
    data_coords.reserve(distance*distance);
    x_stab_coords.reserve((distance*distance)/2);
    z_stab_coords.reserve((distance*distance)/2);
    generate_grid();
    generate_stabilizers();
  }
