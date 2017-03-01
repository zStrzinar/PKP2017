function R = extractTheLargestRegion (T)
%#codegen


% Copy input image and clear it
R = T;
R(:,:,:) = 0;

% Find connected components
CC = replacement_bwconncomp(T, 8);

% Find the largest component
%[ ~, max_idx ] = max([ CC.area ]);
max_idx = -1; max_val = -inf;
for i = 1:numel(CC),
    if CC(i).area > max_val,
        max_val = CC(i).area;
        max_idx = i;
    end
end

if max_idx == -1,
    return;
end

% "Fill" the largest component into empty output image
R(CC(max_idx).pixel_idx) = 1;
end