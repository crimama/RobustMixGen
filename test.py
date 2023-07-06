


from arguments import parser 

if __name__ == '__main__':
    
    config = parser()
    if not config.args.evaluate:
        print('a')
    else:
        pass 
    
