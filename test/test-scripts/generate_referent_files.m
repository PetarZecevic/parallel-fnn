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
## @deftypefn {} {@var{retval} =} generate_referent_files (@var{input1}, @var{input2})
##
## @seealso{}
## @end deftypefn

## Author: Petar Zecevic <pzecevic@pzecevic>
## Created: 2019-01-16

function [] = generate_referent_files (layer_num, start, step, maximum)
  
  input_filename = "files-bin/input";
  output_filename = "files-bin/output";
  weight_filename = "files-bin/weight";
  file_extension = ".bin";
  
  for i = [start:step:maximum]
    # Generate referent input and weights.
    in = randn(i,1);
    w = cell(layer_num,1);
    for j = [1:layer_num]
      w{j} = randn(i);
    endfor
    
    # Write input to file.
    f_name = [input_filename, "-", num2str(i), file_extension];
    fid = fopen(f_name, "w");
    if is_valid_file_id(fid)
      fwrite(fid, in, 'double');
      fclose(fid);
    endif
    
    # Write weights to file.
    f_name = [weight_filename, "-", num2str(i), file_extension];
    fid = fopen(f_name, "w");
    if is_valid_file_id(fid)
      for j = [1:layer_num]
        fwrite(fid, w{j}, 'double');
      endfor
      fclose(fid);
    endif
    
    # Calculate output.
    out = neuralNetworkOutput(in, w, @sigmoidFunc);
    
    # Write output to file.
    f_name = [output_filename, "-", num2str(i), file_extension];
    fid = fopen(f_name, "w");
    if is_valid_file_id(fid)
      fwrite(fid, out, 'double');
      fclose(fid);
    endif
    
  endfor
  
endfunction
