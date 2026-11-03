module DAC8552(
input clk_high,
input [15:0]da,
input update,
input channel,
output reg sync,
output reg sck,
output reg sda
);
reg clk=0;

always@(posedge clk_high)begin
    clk<=~clk;
end

wire [23:0]data={2'b0,channel,~channel,1'b0,channel,2'b0,da};
reg [3:0]clk_cnt=0;
reg flag=0;
reg [4:0]cnt=0;

always@(posedge clk)
    clk_cnt<=clk_cnt+1;
    
wire data_up=clk_cnt==8;
wire clk_up=clk_cnt==0;

always@(posedge clk)begin
    if(update)
        flag<=1;
    else if(cnt==0)
        flag<=0;
end

always@(posedge clk)begin
    if(flag&&clk_up)
        cnt<=cnt-1;
    else if(cnt==0)
        cnt<=26;
    if(flag)begin
        if(clk_up)
            sck<=0;
        else if(data_up)
            sck<=1;
        if(data_up)begin
            if(cnt==25)
                sync<=1;
            else if(cnt==24)
                sync<=0;
            if(cnt&&cnt<25)
                sda<=data[cnt-1];
            else 
                sda<=0;
        end 
    end 
end
endmodule