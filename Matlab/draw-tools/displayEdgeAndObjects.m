function displayEdgeAndObjects(Im, sel_xy, objs, maskedSea, mixture, display_type )

% transform the distribution by rescaling for visualization
T = diag( [ size(Im,2)/size(maskedSea,2), size(Im,1)/size(maskedSea,1)] ) ;
mixture_spatial.Mu = [] ;
mixture_spatial.w = [] ;
mixture_spatial.Cov = {} ;
for i = 1 : 3
   mixture_spatial.Mu = horzcat(mixture_spatial.Mu, T*mixture(i).Mu(1:2,:)) ; 
   mixture_spatial.w = horzcat(mixture_spatial.w, mixture(i).w) ;
   mixture_spatial.Cov = horzcat(mixture_spatial.Cov, T*mixture(i).Cov(1:2,1:2)*T) ;
end
 
% display
switch(display_type)
    case 1
        displayEdgeAndObjects1(Im, sel_xy, objs, maskedSea, mixture_spatial ) ;
    case 2 
        displayEdgeAndObjects2(Im, objs, maskedSea, mixture_spatial ) ;
end
 
% -------------------------------------------------------------------- %
function displayEdgeAndObjects1(Im, sel_xy,objs, maskedSea, mixt_spatial )

figure(1) ; clf ; subplot(1,2,1) ; imagesc(Im) ; hold on ; axis equal ; axis tight
drawDistributionGMM( 'pdf', mixt_spatial, 'color' , [0.5 0.5 0.5] ) ;
hold on ;
if ~isempty(sel_xy)
    plot(sel_xy(1,:),sel_xy(2,:),'g.', 'LineWidth', 2) ;
end

for i_dets = 1 : length(objs)
        rectangle('Position',objs(i_dets).bounding_box,'EdgeColor', 'y', 'LineWidth', 3)  ;
        rectangle('Position',objs(i_dets).bounding_box,'EdgeColor', 'k', 'LineWidth', 2)  ;
end
 
subplot(1,2,2) ; imagesc(maskedSea) ; axis equal ; axis tight ; title('Obstacle map')
subplot(1,2,1) ;
 
% -------------------------------------------------------------------- %
function displayEdgeAndObjects2(Im, objs, maskedSea, mixt_spatial )

maskedSea = imresize(maskedSea, [size(Im,1), size(Im,2)] ,'bilinear') > 0.5 ;
[~, edge_list] = get_edge_list(maskedSea) ;
M2 = Im ; Im2 = Im ;
grIm = rgb2gray(Im) ;
for i = 1 : 3 , M2(:,:,i) = maskedSea ; Im2(:,:,i) = grIm ; end 

clr = [0, 0, 200] ;
w = [0.4, 0.4, 0] ;
for i = 1 : 3
    grIm_b = double(grIm) ;
    grIm_b(maskedSea) = grIm_b(maskedSea)*w(i) + clr(i);    
    Im2(:,:,i) = grIm_b ;
end
 

figure(1) ; clf ; subplot(1,2,1) ; imagesc(Im2) ; hold on ; axis equal ; axis tight ;
a = axis ;
displayComponentsInColor(mixt_spatial) ;
hold on ;
axis(a) ;

subplot(1,2,2) ; imagesc(Im) ; axis equal ; axis tight ; colormap gray ; hold on
for i_dets = 1 : length(objs)
        rectangle('Position',objs(i_dets).BoundingBox,'EdgeColor', 'y', 'LineWidth', 3)  ;
        rectangle('Position',objs(i_dets).BoundingBox,'EdgeColor', 'k', 'LineWidth', 2)  ;
end
plot(edge_list(1,:),edge_list(2,:),'g.', 'LineWidth', 3) ;
subplot(1,2,1) ;

% ------------------------------------------------------------------- %
function displayComponentsInColor(mixt_spatial)

colr = [1,0,0; 0,1,0; 0,0,1];
for i = 1 : 3
    pdf0.Mu = mixt_spatial.Mu(:,i) ;
    pdf0.Cov = mixt_spatial.Cov(i) ;
    pdf0.w = 1 ;
    drawDistributionGMM( 'pdf', pdf0, 'color' , colr(i,:)*0+0.0, 'edgewidth', 3 ) ;
    drawDistributionGMM( 'pdf', pdf0, 'color' , colr(i,:), 'edgewidth', 1 ) ;    
end 

% ------------------------------------------------------------------- %
function [E_out, edgelist] = get_edge_list(M)

E = edge( double(M)*255, 'sobel' )  ;
[L, num] = bwlabel(E, 8) ; 
[nums,~]=hist(L(:),[0:num]) ;
nums = nums(2:end) ; 
[~, cls_id] = max(nums) ;
E_out = zeros(size(E)) ; 
E_out(L==cls_id) = 1 ;
edgelist = regionprops(E_out, 'PixelList') ;
edgelist = edgelist.PixelList' ;
 