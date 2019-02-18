## Copyright (C) 2019 Petar Zecevic
## 
## This program is free software; you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation; either version 3 of the License, or
## (at your option) any later version.
## 
## This program is distributed in the hope that it will be useful,
## but WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
## 
## You should have received a copy of the GNU General Public License
## along with this program.  If not, see <http://www.gnu.org/licenses/>.

## -*- texinfo -*- 
## @deftypefn {} {@var{retval} =} plot_times (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Petar Zecevic <pzecevic@pzecevic>
## Created: 2019-01-16

function [n, vec_serial, vec_parallel] = get_times (serial_time_filename, parallel_time_filename)
  fid = fopen(serial_time_filename, "r");
  if is_valid_file_id(fid)
    [vec_serial, count, err] = fscanf(fid, "%f");
    fclose(fid);
  endif
  
  fid = fopen(parallel_time_filename, "r");
  if is_valid_file_id(fid)
    [vec_parallel, count, err] = fscanf(fid, "%f");
    fclose(fid);
  endif
  
  n = vec_serial(1) : vec_serial(2) : vec_serial(3);
  n = n';
  vec_serial = vec_serial(4:end);
  vec_parallel = vec_parallel(4:end);
  
endfunction
