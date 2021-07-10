import pygame

class pygame_visuals:
    """
    deals with showing text on a pygame window
    functions: self_show_text, show_text, add_message, show_all_messages, show_progress_bar
    """
    
    def __init__(self, gameDisplay, font='arial', fontsize=20, text_color=(255,255,255), topleft=(0,0), center = (300,300), priority = 0):
    
        self.gameDisplay = gameDisplay
        self.font = font
        self.fontsize = fontsize
        self.text_color = text_color
        self.topleft = topleft
        self.center = center
        self.priority = priority
        
        self.messages = []
        
    def show_progress_bar(gameDisplay, percent_complete, topleft = (0,0), length = 100, width = 20):
        
        pygame.draw.rect(gameDisplay, (0,255,0), pygame.Rect(topleft,(length*percent_complete,width)))                                     
        pygame.draw.rect(gameDisplay, (128,128,128), pygame.Rect(topleft,(length,width)), 2)
        
    def add_message(self, topleft = False, gameDisplay = False, font = False, fontsize = False, 
                    text_color = False, antialiasing = True, center = False, priority = False):
    
        if topleft:
            real_topleft = topleft
        else:
            real_topleft = self.topleft
        
        if priority:
            real_priority = priority
        else:
            real_priority = self.priority
                 
        if center:
            real_center = center
        else:
            real_center = self.center
            
        if gameDisplay:
            real_gameDisplay = gameDisplay
        else:
            real_gameDisplay = self.gameDisplay
            
        if font:
            real_font = font
        else:
            real_font = self.font
        
        if fontsize:
            real_fontsize = fontsize
        else:
            real_fontsize = self.fontsize
        
        if text_color:
            real_text_color = text_color
        else:
            real_text_color = self.text_color
            
        self.messages.append([real_topleft, real_gameDisplay, real_font, real_fontsize, real_text_color, 
                              antialiasing, real_center, real_priority])
    
    def show_all_messages(self, *args):
        m = self.messages
        for i in range(len(args)):
            pygame_visuals.show_text(m[i][1], args[i], m[i][0], m[i][2], m[i][3], m[i][4], 
                                     antialiasing = m[i][5], center = m[i][6], priority = m[i][7])

    
    def show_text(gameDisplay, text, topleft = False, font = 'arial', fontsize = 20, 
                  text_color = (255,255,255), return_text_rect = False, antialiasing = True, center = False, priority = 0):
        
        """
        args: self, gameDisplay, text, topleft = False,
                       font = False, fontsize = False, text_color = False, 
                       return_text_rect = False, antialiasing = True
        """
        

        if topleft:
            pass        
        else:
            topleft = (0,0)
                 
        if center:
            pass
        else:
            center = (300,300)
        
        text_font = pygame.font.SysFont(font, fontsize)
        textDisplaySurface = text_font.render(text, antialiasing, text_color)
        text_rect = textDisplaySurface.get_rect()
                 
        if priority:
                 text_rect.center = center
        else: 
                 text_rect.topleft = topleft
        gameDisplay.blit(textDisplaySurface, text_rect)
        
        if return_text_rect:
            return text_rect
        
    def self_show_text(self, text, topleft = False, gameDisplay = False, font = False,
                       fontsize = False, text_color = False, return_text_rect = False, 
                       antialiasing = True, center = False, priority = False):
        """
        args: self, text, topleft = False, gameDisplay = False, 
                       font = False, fontsize = False, text_color = False, 
                       return_text_rect = False, antialiasing = True
        """
        
        if topleft:
            real_topleft = topleft
        else:
            real_topleft = self.topleft
        
        if priority:
            real_priority = priority
        else:
            real_priority = self.priority
                 
        if center:
            real_center = center
        else:
            real_center = self.center
                 
        if gameDisplay:
            real_gameDisplay = gameDisplay
        else:
            real_gameDisplay = self.gameDisplay
            
        if font:
            real_font = font
        else:
            real_font = self.font
        
        if fontsize:
            real_fontsize = fontsize
        else:
            real_fontsize = self.fontsize
        
        if text_color:
            real_text_color = text_color
        else:
            real_text_color = self.text_color
            
        
        text_font = pygame.font.SysFont(real_font, real_fontsize)
        textDisplaySurface = text_font.render(text, antialiasing, real_text_color)
        text_rect = textDisplaySurface.get_rect()
        if real_priority:
                 text_rect.center = real_center
        else:
                text_rect.topleft = real_topleft
        real_gameDisplay.blit(textDisplaySurface, text_rect)
        
        if return_text_rect:
            return text_rect  
        
#################################################################################

class file_io:
    """
    class to help with file parsing. Parses txt files
    functions: from_onedim_list, from_two_dim_list, from_one_dim_nparr, from_two_dim_nparr
    all return lists
    """
    def pygame_rect_to_list(pygame_rect):
        """
        args: pygame_rect
        Note: returns topleft coordinates
        """
        return [pygame_rect.topleft[0], pygame_rect.topleft[1], pygame_rect.size[0], pygame_rect.size[1]]
    
    def from_onedim_list(list_string, dtype = float):
        """
        args: list_string, d_type = float
        Note list_string must include ending braces
        """
        return [dtype(num) for num in list_string[1:-1].split(", ")]
    
    def from_twodim_list(list_string, dtype = float):
        """
        args: list_string, d_type = float
        """
        return[[dtype(num) for num in onedim_list.split(", ")] 
                for onedim_list in list_string[2:-2].split("], [")]
        
    
    def from_onedim_nparr(arr_string, dtype = float):
        """
        args: arr_string, d_type = float
        """
        return [dtype(num) for num in arr_string[1:-1].split(" ") if num != ""]
    
    def from_twodim_nparr(arr_string, dtype = float):
        """
        args: arr_string, d_type = float
        """
        return [[dtype(num) for num in onedim_arr[1:-1].split(" ") if num != ""]
                for onedim_arr in arr_string[1:-1].split("\n ")]