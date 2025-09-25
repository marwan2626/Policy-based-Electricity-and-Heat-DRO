% ---- CONFIG ----
mat_file = 'bev_profiles.mat';      % <-- change to your .mat filename
var_name = 'loadProfiles';       % <-- cell array in the .mat
out_csv  = 'bev_2023_power_first100.csv';

% ---- LOAD ----
S = load(mat_file, var_name);
if ~isfield(S, var_name)
    error('Variable "%s" not found in %s', var_name, mat_file);
end
loadProfiles = S.(var_name);  % Nx1 cell of structs with fields: time, power, soc

nCarsTotal = numel(loadProfiles);
if nCarsTotal == 0
    error('"%s" is empty.', var_name);
end

% Determine series length from the first entry
first = loadProfiles{1};
if ~isstruct(first) || ~all(isfield(first, {'time','power','soc'}))
    error('Cells of "%s" must be structs with fields time,power,soc.', var_name);
end
N = numel(first.power);  % should be 35040 for 15-min steps over a non-leap year

% Number of cars to export (cap at available)
K = min(100, nCarsTotal);

% Build datetime vector for 2023 at 15-min resolution
dt = datetime(2023,1,1,0,0,0) + minutes((0:N-1)'*15);

% ---- OPEN OUTPUT & WRITE HEADER ----
fid = fopen(out_csv, 'w');
if fid == -1, error('Could not open %s for writing.', out_csv); end

% Header
fprintf(fid, 'datetime');
for c = 1:K
    fprintf(fid, ',car_%d', c);
end
fprintf(fid, '\n');

% ---- STREAM ROWS ----
% For each time index i, write timestamp then power(i) for cars 1..K
for i = 1:N
    % Write datetime in ISO-like format
    % (Using datestr for compatibility; you can also use string(dt(i),'yyyy-MM-dd HH:mm'))
    fprintf(fid, '%s', datestr(dt(i), 'yyyy-mm-dd HH:MM'));
    
    for c = 1:K
        p = loadProfiles{c}.power(i);
        fprintf(fid, ',%.10g', p);
    end
    fprintf(fid, '\n');
end

fclose(fid);
fprintf('Done. Wrote %s with %d rows and %d columns (datetime + %d cars)\n', out_csv, N, 1+K, K);
