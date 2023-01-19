%% getUniformPatterns
function uniformPatterns = getUniformPatterns() 
    
    % in decimal, from 1 to 256
    uniformPatterns = zeros(1,58,'int16');
    count = 1;
 
    for i= 1 : 256     
        value = dec2bin(i-1,8);
    
        count0 = extract(value, '01');
        count1 = extract(value, '10');
          
        if size(count0,1) + size(count1,1) <= 2 
            value = bin2dec(value) + 1;
            uniformPatterns(count) = value;
            count = count +1;
        end
    end
end
