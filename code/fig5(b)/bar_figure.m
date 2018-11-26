lineColor = linspecer(7);

f1=[0.2826; 0.15; 0; 0     ; 0   ; 0; 0     ; 0     ; 0; 0   ; 0     ; 0; 0  ; 0     ; 0; 0     ; 0     ; 0; 0     ; 0   ];
f2=[0     ; 0   ; 0; 0.2500; 0.18; 0; 0     ; 0     ; 0; 0   ; 0     ; 0; 0  ; 0     ; 0; 0     ; 0     ; 0; 0     ; 0   ];
f3=[0     ; 0   ; 0; 0     ; 0   ; 0; 0.3953; 0.1860; 0; 0   ; 0     ; 0; 0  ; 0     ; 0; 0     ; 0     ; 0; 0     ; 0   ];
f4=[0     ; 0   ; 0; 0     ; 0   ; 0; 0     ; 0     ; 0; 0.25; 0.1875; 0; 0  ; 0     ; 0; 0     ; 0     ; 0; 0     ; 0   ];
f5=[0     ; 0   ; 0; 0     ; 0   ; 0; 0     ; 0     ; 0; 0   ; 0     ; 0; 0.2; 0.1667; 0; 0     ; 0     ; 0; 0     ; 0   ];
f6=[0     ; 0   ; 0; 0     ; 0   ; 0; 0     ; 0     ; 0; 0   ; 0     ; 0; 0  ; 0     ; 0; 0.1429; 0.1429; 0; 0     ; 0   ];
f7=[0     ; 0   ; 0; 0     ; 0   ; 0; 0     ; 0     ; 0; 0   ; 0     ; 0; 0  ; 0     ; 0; 0     ; 0     ; 0; 0.2264; 0.13];

x = 1:length(f1);

% Plot the data.
h1 = bar(f1, 'FaceColor', lineColor(1, :));hold on
h2 = bar(f2, 'FaceColor', lineColor(2, :));hold on
h3 = bar(f3, 'FaceColor', lineColor(3, :));hold on
h4 = bar(f4, 'FaceColor', lineColor(4, :));hold on
h5 = bar(f5, 'FaceColor', lineColor(5, :));hold on
h6 = bar(f6, 'FaceColor', lineColor(6, :));hold on
h7 = bar(f7, 'FaceColor', lineColor(7, :));
%h = bar(f1);hold on
%h = bar(f2)
% Reduce the size of the axis so that all the labels fit in the figure.
pos = get(gca,'Position');
set(gca,'Position',[pos(1), .2, pos(3) .65])

% Add a title, if you need it.
%title('')

% Set X-tick positions
Xt = x;

% If you want to set x-axis limit, uncomment the following two lines of 
% code and remove the third
Xl = [0 21]; 
set(gca,'XTick', Xt, 'XLim',Xl);
%set(gca,'XTick', Xt);
% ensure that each string is of the same length, using leading spaces
algos = ['   Drama';'  Action'; '        '; '   Drama'; '  Comedy'; '        '; '  Comedy'; ' Romance'; '        '; '   Drama'; '  Comedy'; '        ';
    'Thriller'; '   Drama'; '        ';'  Action'; '  Comedy'; '        ';' Romance'; '  Action'];

ax = axis; % Current axis limits
axis(axis); % Set the axis limit modes (e.g. XLimMode) to manual
Yl = ax(3:4); % Y-axis limits

% Remove the default labels
set(gca,'XTickLabel','')

% Place the text labels
t = text(Xt,Yl(1)*ones(1,length(Xt)), algos(1:length(f1),:));

set(t,'HorizontalAlignment','right','VerticalAlignment','top', ...
'Rotation', 60, 'Fontsize', 13);
ylabel('Y-Axis Label', 'Fontsize', 13)

%% you don't have to run the following lines, if you don't need xlabel
% Get the Extent of each text object. This
% loop is unavoidable.
for i = 1:length(t)
ext(i,:) = get(t(i),'Extent');
end

% Determine the lowest point. The X-label will be
% placed so that the top is aligned with this point.
LowYPoint = min(ext(:,2));

% Place the axis label at this point
XMidPoint = Xl(1)+abs(diff(Xl))/2;
tl = text(XMidPoint,LowYPoint,'X-Axis Label', ...
'VerticalAlignment','top', ...
'HorizontalAlignment','center');

legend([h1, h2, h3, h4, h5, h6, h7], 'Under 18', '18-24', '25-34', '35-44', '45-49', '50-55', '56+');