def remove_n_t_xad(t) -> str: 
    return t.replace("\n", "").replace("\xad", "").replace("\t", "")

def remove_extra_links(t) -> str: 
    return t.replace("http://eduportal.uz", "")

def selfwork(t) -> str:
    if "САМОСТОЯТЕЛЬНАЯ РАБОТА" in t: 
        return ''
    elif "Г Л А В Л Е Н И Е" in t or "Р А З Д Е Л " in t: 
        return ''
    else: 
        return t

def prepare(t) -> str:
    t = selfwork(t)
    t = remove_extra_links(t)
    t = remove_n_t_xad(t)
    
    return t 

